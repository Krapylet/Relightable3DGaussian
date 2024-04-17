#include "shader.h"
#include "config.h"
#include <cooperative_groups.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace CudaShader
{
    __device__ shaderParams::shaderParams(PackedShaderParams p, int idx):
        W(p.W),
        H(p.H),
		orig_point({p.orig_points[idx + 0], p.orig_points[idx + 1], p.orig_points[idx + 2]}),
        // world space position = 
        // view space up, right, forward,
        // world space up, right, forward,  		
		point_xy_image(p.points_xy_image[idx].x, p.points_xy_image[idx].y),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix (p.projmatrix),
		projmatrix_inv (p.projmatrix_inv),
		focal_x (p.focal_x),
        focal_y (p.focal_y),
		tan_fovx (p.tan_fovx), 
        tan_fovy (p.tan_fovy),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),

		// pr. frame texture information
		depth (p.depths[idx]),                       
		color ({p.colors[idx + 0], p.colors[idx + 0], p.colors[idx + 0]}),	// TODO: merge with out_color		
		conic_opacity ({p.conic_opacity[idx].x, p.conic_opacity[idx].y, p.conic_opacity[idx].z, p.conic_opacity[idx].w}), // Todo: split opacity to own variable

		// Precomputed 'texture' information from the neilf pbr decomposition
		brdf_color ({p.features[idx + 0], p.features[idx + 1], p.features[idx + 2]}),
		normal ({p.features[idx + 3], p.features[idx + 4], p.features[idx + 5]}),
		base_color ({p.features[idx + 6], p.features[idx + 7], p.features[idx + 8]}),
		roughness (p.features[idx + 9]),
		metallic (p.features[idx + 10]),
		incident_light (p.features[idx + 11]),
		local_incident_light (p.features[idx + 12]),
		global_incident_light (p.features[idx + 13]),
		incident_visibility (p.features[idx + 14]),

		// output
		// We use pointers to the output instead of return values to make it easy to extend during development.
        out_color (p.out_color + idx * NUM_CHANNELS)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    template<int C>
    __device__ static void OutlineShaderCUDA(shaderParams p)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = p.camera_position - p.orig_point;
        float angle = 1 - abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(p.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        // Set output color
        //TODO: Make into glm::vec3 or something.
        p.out_color[0] = p.color.r * opacity;
        p.out_color[1] = p.color.g * opacity;
        p.out_color[2] = p.color.b * opacity;
    }

    template<int C>
    __device__ static void WireframeShaderCUDA(shaderParams p)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = p.camera_position - p.orig_point;
        float angle = 1 - abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(p.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        // Set output color
        p.out_color[0] = 1 - opacity;
        p.out_color[1] = 1 - opacity;
        p.out_color[2] = 1 - opacity;
    }

    ///// Assign all the shaders to their short handles.
    __device__ shader outlineShader = &OutlineShaderCUDA<NUM_CHANNELS>;
    __device__ shader wireframeShader = &WireframeShaderCUDA<NUM_CHANNELS>;

    __global__ void ExecuteShader(shader shader, PackedShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.splatsInShader)
            return;
        idx += packedParams.shaderStartingOffset;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        shaderParams params(packedParams, idx);

        // No need to dereference the shader function pointer.
        shader(params);
    }

}

