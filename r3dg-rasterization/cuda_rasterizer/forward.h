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
#include "../utils/texture.h"
#include "postProcessShader.h"
#include "../shaderManager.h"

#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace FORWARD
{

	// Run any user-provided SH shaders prior to preprocessing, allowing users to change positions and such.
	void RunSHShaders(
		const int P,
		ShaderManager* h_shShaderManager,

		//input
		float const time, float const dt,
		float const scale_modifier,
		float const *const viewmatrix,
		float const *const viewmatrix_inv,
		float const *const projmatrix,
		float const *const projmatrix_inv,
		int const W, int const H,
		float const focal_x, float const focal_y,
		float const tan_fovx, float const tan_fovy,
		int deg, int max_coeffs,
		int const S,
		float *const features,
		Texture::TextureManager *const d_textureManager,

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const positions,
		glm::vec3 *const scales,
		glm::vec4 *const rotations,
		float *const opacities,
		glm::vec3 *const shs,

		// output
		float *const stencil_vals
		);

	// Perform initial steps for each Gaussian prior to rasterization.
	void PreProcess(int P, int D, int M,
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

	void RenderIntermediateTextures(
		dim3 tile_grid, dim3 block,
		const uint2* __restrict__ ranges,
		const uint32_t* __restrict__ point_list,
		int W, int H,
		const float2* __restrict__ screen_positions,
		const float* __restrict__ depths,
		const float* __restrict__ stencils,
		const float4* __restrict__ conic_opacity,
		float* __restrict__ out_depth,
		float* __restrict__ out_stencil
		);

	// Runs after preprocess but before renderer. Allows changing rgb output for individual splats.
	void RunSplatShaders(
		int const P,
		ShaderManager* h_splatShaderManager,

		// intput
		int const W, int const H,
		float const time, float const dt,
		float const *const positions,  
		float2 const *const screen_positions,
		float const *const depth_tex,
		float const *const stencil_tex,
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
		float *const features,
		Texture::TextureManager *const d_textureManager,

		// input/output
		float *const stencils,

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

	void RunPostProcessShaders(
		std::vector<PostProcess::PostProcessShader> postProcessPasses,

		int const width, int const height,
		float const time, float const dt,
		const float * const __restrict__ viewmatrix,
		const float * const __restrict__ viewmatrix_inv,
		const float * const __restrict__ projmatrix,
		const float * const __restrict__ projmatrix_inv,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float const * const __restrict__ background,
		float const * const __restrict__ out_color,
		float const * const __restrict__ out_opacity,
		float const * const __restrict__ out_depth,
		float const * const __restrict__ stencil_tex,
		int const S,
		float const *const __restrict__ out_feature,
		Texture::TextureManager * const d_textureManager,

		float * const __restrict__ out_shader_color
	);

	
}

#endif