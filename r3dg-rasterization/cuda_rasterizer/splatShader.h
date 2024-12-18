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

#ifndef GLM_FORCE_CUDA
	#define GLM_FORCE_CUDA
#endif

#include <glm/glm.hpp>

namespace SplatShader
{
	// Encapsulate shader parameters in a struct so it becomes easy to update during development.
	// This representation contains data for all the splats packed together.
	struct PackedSplatShaderParams {
		// shader execution information:
		int const P;						// Total number of splats.

		// Screen information:
        int const W; int const H;			

		// Time information
		float const time; float const dt;

		// position information
		glm::vec3 const *const __restrict__ positions;  			
		glm::vec2 const *const __restrict__ screen_positions;

		// Screen texture information. Indexed with floor(screen_pos.x) + floor(screen_pos.y) * screen.width
		float const * const depth_tex;
		float const * const stencil_tex;

		// Projection information.
		float const *const __restrict__ viewmatrix;
		float const *const __restrict__ viewmatrix_inv;
		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;

		// pr. frame texture information
		float const *const __restrict__ splat_depths;				
		glm::vec3 const *const __restrict__ colors_SH;				
		glm::vec4 *const __restrict__ conic_opacity;		

		// Precomputed 'texture' information from the R3DG BDRF decomposition
		int const  S;						// Feature channel count.
		float *const __restrict__ features;		// Interleaved array of precomputed 'textures' for each individual gaussian. Stored in the following order:
											// float  roughness,
                                            // float  metallic
                                            // float  incident_visibility
                                            // float3 brdf_color,
                                            // float3 normal,	       Object space
                                            // float3 base_color,
                                            // float3  incident_light
                                            // float3  local_incident_light
                                            // float3  global_incident_light

		Texture::TextureManager *const d_textureManager;

		// input / output
		float *const __restrict__ stencils;			//The stencil value of the individual splat
		float *const stencil_opacities;				//A seperate opacity that is used when rendering the stencil mask

		// output
		glm::vec3 *const __restrict__ out_colors;			// shader color output.
	};

	// --------- Shader input and output interfaces.
	// Only contains information relevant to each individual splat.
	// Acts as an interface layer that hides complexities and calculates commonly used values, thereby reducing boiler-plate code and human error.

	// Inputs that cannot be changed by the shader
	struct SplatShaderConstantInputs {
		// Constructor
		__device__ SplatShaderConstantInputs(PackedSplatShaderParams params, int idx);

		// Screen information:
        int const W; int const H;							// Sceen width and height

		// Time information:
		float const time; 				// Time since program start in ms
		float const dt;					// Time since last frame in ms

        // position information:
		glm::vec3 const position;  				// mean 3d position of gaussian in world space. Can't be changed since 2D screen position has already been calculated.
		glm::vec2 const screen_position;		// mean 2d position of gaussian in screen space.
		int const mean_pixel_idx; 				// Index of the mean pixel position of the splat.

		// Screen texture information. Indexed with mean_pixel_idx
		float const * const depth_tex;			// Scene depth texture. Indexed with mean_pixel_idx
		float const * const stencil_tex;		// Stencil_tex depth texture. Indexed with mean_pixel_idx

		// Projection information.
		// RightX  RightY  RightZ  0
		// UpX     UpY     UpZ     0
		// LookX   LookY   LookZ   0
		// PosX    PosY    PosZ    1
		float const *const __restrict__ viewmatrix;

		// RightX  UpX     LookX      0
		// RightY  UpY     LookY      0
		// RightZ  UpZ     LookZ      0
		// -(Pos*Right)  -(Pos*Up)  -(Pos*Look)  1
		float const *const __restrict__ viewmatrix_inv;

		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
		float const focal_x;				// Camera horizontal focal length
		float const focal_y;				// Camera vertical focal length	
		float const tan_fovx;				// Camera horizontal field of vision	
		float const tan_fovy;				// Camera vertical field of vision		
		glm::vec3 const camera_position;	// Position of camera in world space

		// pr. frame splat information
		float const splat_depth;						// Mean splat depth in view space.
		glm::vec3 const conic;							// Covariance matrix used to determine splats shapes in the final rendering step.
		glm::vec3 const *const __restrict__ color_SH;	// Color from SH evaluation

		// Class that is used to retrieve textures. Make sure to cache textures once retrieved.
		Texture::TextureManager const *const d_textureManager;
	};

	// Modifiable intputs
	// Inputs values that can be changed by the shader
	// We use pointers to the outputs/modifiable inputs so we can operate in-place.
	struct SplatShaderModifiableInputs {
		// Constructor
		__device__ SplatShaderModifiableInputs(PackedSplatShaderParams params, int idx);

		// Precomputed 'texture' information from the R3DG BDRF decomposition
		glm::vec3 *const color_brdf;			// pbr splat color
		glm::vec3 *const normal;				// Splat normal in object space
		glm::vec3 *const color_base;			// Decomposed splat color without lighting
		float *const  roughness;
		float *const  metallic;
		glm::vec3 *const  incident_light;			// total amout of light hitting this 3D Gaussian
		glm::vec3 *const  local_incident_light;		// bounce light that hits this 3D gaussian
		glm::vec3 *const  global_incident_light;	// Global light that hits this 3D gaussian
		float *const  incident_visibility;			// Fraction of how much global light hits this 3D Gaussian

		// pr. splat information
		float *const __restrict__ opacity;				//The opacity of the splat. Opacity works a bit funky because how splats are blended. It is better to multiply this paramter by something rather than setting it to specific values.
		float *const __restrict__ stencil_val;			//The stencil value of the splat.
		float *const stencil_opacity;					//A seperate opacity that is used when rendering the stencil mask
	};

	// output
	struct SplatShaderOutputs{
		// Constructor
		__device__ SplatShaderOutputs(PackedSplatShaderParams params, int idx);

		glm::vec3 *const __restrict__ out_color;					// RGB color output the splat. Will get combined based on alpha and depth in the next step.
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*SplatShader)(SplatShaderConstantInputs in, SplatShaderModifiableInputs io, SplatShaderOutputs out);

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind
	std::map<std::string, int64_t> GetSplatShaderAddressMap();

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteSplatShaderCUDA(SplatShader shader, int* d_splatIndexes, PackedSplatShaderParams packedParams);

	
};
