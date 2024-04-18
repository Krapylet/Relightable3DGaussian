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
		const float* positions,
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
		float2* screen_positions,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);


	// Runs after preprocess but before renderer. Allows changing values for individual gaussians.
	void shade(
		int const shaderCount,
		float const *const shaderIDs,			
		float const *const shaderIndexOffset,
		int const W, int const H,	
		int const P,
		float const *const positions,  
		float2 const *const screen_positions,
		float const *const viewmatrix,
		float const *const viewmatrix_inv,
		float const *const projmatrix,
		float const *const projmatrix_inv,
		const float focal_x, float const focal_y,
		const float tan_fovx, float const tan_fovy,
		float const *const depths,		
		float const *const colors_SH, 
		float4 const *const conic_opacity,        
		int const S,						
		float const *const features,	
		// output
		float *const out_colors
	);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int S, int W, int H,
		const float2* screen_positions,
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