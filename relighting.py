import json
import os
import cv2
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.cameras import Camera
from scene.envmap import EnvLight
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from tqdm import tqdm
from bvh import RayTracer
from gaussian_renderer.neilf_composite import sample_incident_rays
import asset_processing.textureImport as texImport
import asset_processing.PostProcess as PostProcess


def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def scene_composition(scene_dict: dict, dataset: ModelParams):
    gaussians_list = []
    for scene in scene_dict:
        gaussians = GaussianModel(dataset.sh_degree, render_type="neilf")
        gaussians.load_ply(scene_dict[scene]["path"])

        torch_transform = torch.tensor(scene_dict[scene]["transform"], device="cuda").reshape(4, 4)
        gaussians.set_transform(transform=torch_transform)

        gaussians_list.append(gaussians)

    gaussians_composite = GaussianModel.create_from_gaussians(gaussians_list, dataset)
    n = gaussians_composite.get_xyz.shape[0]
    print(f"Totally {n} points loaded.")

    gaussians_composite._visibility_rest = (
        torch.nn.Parameter(torch.cat(
            [gaussians_composite._visibility_rest.data,
             torch.zeros(n, 5 ** 2 - 4 ** 2, 1, device="cuda", dtype=torch.float32)],
            dim=1).requires_grad_(True)))

    gaussians_composite._incidents_dc.data[:] = 0
    gaussians_composite._incidents_rest.data[:] = 0

    return gaussians_composite


def update_visibility(gaussians, is_bake=True):
    if is_bake:
        gaussians.finetune_visibility(iterations=1000)
    else:
        raytracer = RayTracer(gaussians.get_xyz, gaussians_composite.get_scaling,
                              gaussians.get_rotation)
        gaussians_xyz = gaussians.get_xyz
        gaussians_inverse_covariance = gaussians.get_inverse_covariance()
        gaussians_opacity = gaussians.get_opacity[:, 0]
        gaussians_normal = gaussians.get_normal
        incident_visibility_results = []
        chunk_size = gaussians_xyz.shape[0] // ((args.sample_num - 1) // 24 + 1)
        for offset in tqdm(range(0, gaussians_xyz.shape[0], chunk_size),
                           "Precompute raytracing visibility"):
            incident_dirs, _ = sample_incident_rays(gaussians_normal[offset:offset + chunk_size], False,
                                                    args.sample_num)
            trace_results = raytracer.trace_visibility(
                gaussians_xyz[offset:offset + chunk_size, None].expand_as(incident_dirs),
                incident_dirs,
                gaussians_xyz,
                gaussians_inverse_covariance,
                gaussians_opacity,
                gaussians_normal)
            incident_visibility = trace_results["visibility"]
            incident_visibility_results.append(incident_visibility)
        incident_visibility_result = torch.cat(incident_visibility_results, dim=0)
        gaussians._visibility_tracing = incident_visibility_result

    return gaussians


def render_points(camera, gaussians):
    intrinsic = camera.get_intrinsics()
    w2c = camera.world_view_transform.transpose(0, 1)

    xyz = gaussians.get_xyz
    print(f"Xyz shape: {xyz.shape}\n")
    color = gaussians.get_base_color
    print(f"color shape: {color.shape}\n")
    xyz_homo = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]
    z = xyz_cam[:, 2]
    uv_homo = xyz_cam @ intrinsic.T
    uv = uv_homo[:, :2] / uv_homo[:, 2:]
    uv = uv.long()
    
    valid_H = torch.logical_and(uv[:, 1] >= 0, uv[:, 1] < H)
    print(valid_H)
    valid_W = torch.logical_and(uv[:, 0] >= 0, uv[:, 0] < W)
    print(valid_W)

    valid_point = torch.logical_and(valid_H, valid_W)

    uv = uv[valid_point]
    z = z[valid_point]
    color = color[valid_point]

    depth_buffer = torch.full_like(render_pkg['render'][0], 10000)
    rgb_buffer = torch.full_like(render_pkg['render'], bg)
    while True:
        mask = depth_buffer[uv[:, 1], uv[:, 0]] > z
        if mask.sum() == 0:
            break
        uv_mask = uv[mask]
        depth_buffer[uv_mask[:, 1], uv_mask[:, 0]] = z[mask]
        rgb_buffer[:, uv_mask[:, 1], uv_mask[:, 0]] = color[mask].transpose(-1, -2)

    return rgb_buffer


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument('-co', '--config', default=None, required=True, help="the config root")
    parser.add_argument('-e', '--envmap_path', default=None, help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=None,
                        help="If set, use it as background color")
    parser.add_argument('--bake', action='store_true', default=False, help="Bake the visibility and refine.")
    parser.add_argument('--video', action='store_true', default=False, help="If True, output video as well.")
    parser.add_argument('--output', default="./capture_trace", help="Output dir.")
    parser.add_argument('--capture_list',
                        default="base_color, metallic, normal, pbr ,pbr_env, points, render, roughness, visibility",
                        help="what should be rendered for output.")
    args = parser.parse_args()
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    # load configs
    scene_config_file = f"{args.config}/transform.json"
    traject_config_file = f"{args.config}/trajectory.json"
    light_config_file = f"{args.config}/light_transform.json"

    scene_dict = load_json_config(scene_config_file)
    traject_dict = load_json_config(traject_config_file)
    light_dict = load_json_config(light_config_file)

    # load gaussians
    light = EnvLight(path=args.envmap_path, scale=1)
    gaussians_composite = scene_composition(scene_dict, dataset)

    # update visibility
    gaussians_composite = update_visibility(gaussians_composite, args.bake)

    # rendering
    capture_dir = args.output
    os.makedirs(capture_dir, exist_ok=True)
    capture_list = [str.strip() for str in args.capture_list.split(",")]
    for capture_type in capture_list:
        capture_type_dir = os.path.join(capture_dir, capture_type)
        os.makedirs(capture_type_dir, exist_ok=True)

    bg = args.background_color
    if bg is None:
        bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict['neilf_composite']

    # Import textures used in shaders
    textureImporter = texImport.TextureImporter()
    d_textureManager_ptr = textureImporter.initialize_all_textures()

    # Tell each splat which shader to use
    gaussians_composite.append_shader_addresses_gpu_accelerated()

    #Select which post processing passes.
    postProcessManager = PostProcess.PostProcessManager()
    #postProcessManager.AddPostProcessingPass("crackReconstriction")

    render_kwargs = {
        "pc": gaussians_composite,
        "pipe": pipe,
        "bg_color": background,
        "is_training": False,
        "dict_params": {
            "env_light": light,
            "sample_num": args.sample_num,
        },
        "bake": args.bake,
        "d_textureManager_ptr": d_textureManager_ptr,
        "postProcessingPasses" : postProcessManager.activePostProcessingPasses,
    }

    H = traject_dict["camera"]["height"]
    W = traject_dict["camera"]["width"]
    fovx = traject_dict["camera"]["fov"] * np.pi / 180
    fovy = focal2fov(fov2focal(fovx, W), H)

    progress_bar = tqdm(traject_dict["trajectory"].items(), desc="Rendering")
    for idx, cam_info in progress_bar:
        w2c = np.array(cam_info, dtype=np.float32).reshape(4, 4)

        R = w2c[:3, :3].T
        T = w2c[:3, 3]
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)

        if light_dict is not None:
            light.transform = torch.tensor(light_dict["transform"][idx], dtype=torch.float32, device="cuda").reshape(3,
                                                                                                                     3)

        with torch.no_grad():
            render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

        for capture_type in capture_list:
            if capture_type == "points":
                render_pkg[capture_type] = render_points(custom_cam, gaussians_composite)
            if capture_type == "normal":
                render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
            elif capture_type in ["base_color", "roughness", "metallic", "visibility"]:
                print(f"{capture_type}: shape {render_pkg[capture_type].shape}, Opacity shape: {render_pkg['opacity'].shape}\n")
                print(f"Opacoty * bg shape: {((1 - render_pkg['opacity']) * bg).shape}\n")
                render_pkg[capture_type] = render_pkg[capture_type] + (1 - render_pkg['opacity']) * bg
            save_image(render_pkg[capture_type], f"{capture_dir}/{capture_type}/frame_{idx}.png")

    # output as video
    if args.video:
        progress_bar = tqdm(capture_list, desc="Outputting video")
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        for capture_type in progress_bar:
            video_path = f"{capture_dir}/{capture_type}.mp4"
            image_names = [os.path.join(capture_dir, capture_type, f"frame_{j}.png") for j in
                           traject_dict["trajectory"].keys()]
            media_writer = cv2.VideoWriter(video_path, fourcc, 60, (W, H))

            for image_name in image_names:
                img = cv2.imread(image_name)
                media_writer.write(img)
            media_writer.release()
