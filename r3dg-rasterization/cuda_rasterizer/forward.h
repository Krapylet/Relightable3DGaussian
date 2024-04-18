/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);


	// Runs after preprocess but before renderer. Allows changing values for individual gaussians.
	void shade(
		const int shaderCount,
		const float* shaderIDs,			
		const float* shaderSplatCount,  // Number of splats to render with each shader
		int W, int H,	
		// TODO:  void *shader			// Function pointer to specific shader to call.
		// Gaussian information:
		int P,							// Total number of gaussians.
		const float* orig_points,  		// mean 3d position of gaussian in world space.
		float2* points_xy_image,		// mean 2d position of gaussian in screen space.
		// Projection information
		const float* viewmatrix,
		const float* viewmatrix_inv,
		const float* projmatrix,
		const float* projmatrix_inv,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		// pr. frame texture information
		float* depths,					// Gaussian depth in view space.
		float* colors,
		float4* conic_opacity,          // ???? Read up on original splatting paper.
		// Precomputed 'texture' information
		int S,							// Feature channel count.
		const float* features,			// Interleaved array of precomputed 'textures', such as color, normal, AOO ect for each gaussian.
		// output
		float* shader_colors					// Raw Gaussian SH color.
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int S, int W, int H,
		const float2* points_xy_image,
		const float* depths,
		const float* features,
		const float* shader_colors,
		const float* colors,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_opacity,
		float* out_depth,
		float* out_feature,
		float* out_shader_color);

	void render_xyz(
        const dim3 grid, dim3 block,
		const int W, const int H,
		const float* viewmatrix,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float tan_fovx, const float tan_fovy,
		const float* opacities,
		const float* depths,
		float* normals,
		float* surface_xyz);

	void render_pseudo_normal(
        const dim3 grid, dim3 block,
		const int W, const int H,
		const float* viewmatrix,
		const float focal_x, const float focal_y,
		const float cx, const float cy,
		const float tan_fovx, const float tan_fovy,
		const float* opacities,
		const float* depths,
		float* normals,
		float* surface_xyz);

	
}

#endif