/*
*  Inspired by lai Mao at https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
*  And the official cuda sample at https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/FunctionPointers/FunctionPointers_kernels.cu
*
*
*/

#pragma once

#include <cuda.h>
#include "cuda_runtime.h"
#include <vector>
#include <functional>
#include <string>
#include <map>
#include "../utils/texture.h"
#include "../utils/indirectMap.h"

#ifndef GLM_FORCE_CUDA
	#define GLM_FORCE_CUDA
#endif

#include <glm/glm.hpp>

namespace SplatShader
{
	// Encapsulate shader parameters in a struct so it becomes easy to update during development.
	// This representation contains data for all the splats packed together.
	struct PackedSplatShaderParams {
		// Screen information:
        int const W; int const H;			

        // shader execution information:
		int const P;						// Total number of splats.

		// Time information
		float const time; float const dt;

		// position information
		glm::vec3 const *const __restrict__ positions;  			
		glm::vec2 const *const __restrict__ screen_positions;		

		// Projection information.
		float const *const __restrict__ viewmatrix;
		float const *const __restrict__ viewmatrix_inv;
		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;

		// pr. frame texture information
		float const *const __restrict__ depths;				
		glm::vec3 const *const __restrict__ colors_SH;				
		glm::vec4 const *const __restrict__ conic_opacity;		

		// Precomputed 'texture' information from the neilf pbr decomposition
		int const  S;						// Feature channel count.
		float const *const __restrict__ features;		// Interleaved array of precomputed 'textures' for each individual gaussian. Stored in the following order:
                                            // float3 brdf_color,
                                            // float3 normal,	       Object space
                                            // float3 base_color,
                                            // float  roughness,
                                            // float  metallic
                                            // float  incident_light
                                            // float  local_incident_light
                                            // float  global_incident_light
                                            // float  incident_visibility

		Texture::TextureManager *const d_textureManager;

		// output
		// In producion code, the colors field should function both as SH color input and as color output though reassignment, but we keep them seperate to make it easy to illustrate the difference.
		glm::vec3 *const __restrict__ out_colors;			// shader color output.
	};

	// Used as input and output interface to the shaders.
	// Only contains information relevant to each individual splat.
	// Acts as an interface layer that hides complexities, calculate commonly used values, thereby reducing boiler-plate code and human error.
	// The reason we don't just create unpacked params from the start, is that it would take too long to do in the host functions.
	//TODO: Test memory and speed cost of this approach.
	//TODO: Also pass SHs? Can we do something interesting with them in the code? They function as a low-pass filter on the detail if you reduce the order.
	struct SplatShaderParams {
		// Constructor
		__device__ SplatShaderParams(PackedSplatShaderParams params, int idx);

		// Screen information:
        int const W; int const H;							// Sceen width and height
		// TODO: Collapse depth into a screen texture during the preprocessing (after the SH shader), so we can see the depth of the entire scene during this step.
		// This woudl be a cheap way to approximate how visible each individual splat is. 

		// Time information:
		float const time; float const dt;

        // position information:
		glm::vec3 const position;  			// mean 3d position of gaussian in world space.
		glm::vec2 const screen_position;		// mean 2d position of gaussian in screen space.

		// Projection information.
		float const *const __restrict__ viewmatrix;
				// RightX  RightY  RightZ  0
                // UpX     UpY     UpZ     0
                // LookX   LookY   LookZ   0
                // PosX    PosY    PosZ    1
		float const *const __restrict__ viewmatrix_inv;
				// RightX  UpX     LookX      0
                // RightY  UpY     LookY      0
                // RightZ  UpZ     LookZ      0
                // -(Pos*Right)  -(Pos*Up)  -(Pos*Look)  1
		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;
		glm::vec3 const camera_position;	// Position of camera in world space

		// pr. frame texture information
		float const depth;					// Mean splat depth in view space.
		glm::vec4 const conic_opacity;		// ???? Float4 that contains conic something in the first 3 indexes, and opacity in the last. Read up on original splatting paper.
		glm::vec3 const *const __restrict__ color_SH;	// Color from SH evaluation

		// Precomputed 'texture' information from the neilf pbr decomposition
		glm::vec3 const color_brdf;			// pbr splat color
		glm::vec3 const normal;				// Splat normal in object space
		glm::vec3 const color_base;			// Decomposed splat color without lighting
		float const  roughness;
		float const  metallic;
		float const  incident_light;
		float const  local_incident_light;
		float const  global_incident_light;
		float const  incident_visibility;

		// Class that is used to retrieve textures. Make sure to cache textures once retrieved.
		Texture::TextureManager *const d_textureManager;

		// output
		// We use pointers to the output instead of return values to make it easy to extend during development.
		glm::vec3 *const __restrict__ out_color;					// RGB color output the splat. Will get combined based on alpha in the next step.
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
	//typedef std::function<void(SplatShaderParams)> SplatShader;
    typedef void (*SplatShader)(SplatShaderParams params);

	// Returns a map of shader names and shader device function pointers
	IndirectMap<char*, SplatShader>* GetSplatShaderAddressMap();

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(SplatShader*, PackedSplatShaderParams packedParams);

	
};
