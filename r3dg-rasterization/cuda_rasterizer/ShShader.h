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

namespace ShShader
{
	struct PackedShShaderParams {
        const int P;

		// Screen information
		int const W; int const H;

        //time
		float const time; float const dt;
		
		// global splat parameters
		float const scale_modifier;
        int deg; int max_coeffs;

		// Projection information
		float const *const viewmatrix;
		float const *const viewmatrix_inv;
		float const *const projmatrix;
		float const *const projmatrix_inv;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;

		// pbr decomposition "textures"
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

		// textures
		Texture::TextureManager *const d_textureManager;

		// geometry
		glm::vec3 *const positions;
		glm::vec3 *const scales;
		glm::vec4 *const rotations;

		// SH color  
		glm::vec3 *const shs;
		float *const opacities;

		// intermediate textures
		float *const stencil_vals;
		float *const stencil_opacities;
	};

	struct ShShaderConstantInputs {
		// Constructor
		__device__ ShShaderConstantInputs(PackedShShaderParams params, int idx);

		// Screen information
		int const W; 				// Width of the screen in pixels
		int const H;				// height of the screen in pixels

        //time
		float const time; 			// Time since program start in ms
		float const dt;				// Time since last frame in ms

		// Global splat parameters
		float const scale_modifier;		// How much each splat needs to be scaled to match the scene size
		int const deg;				// Numer of active SH bands
		int const max_coeffs;		// Total number of SH coefficients
			
		// projection information
		float const *const viewmatrix;				// The scene's viewmatrix
		float const *const viewmatrix_inv;			// The scene's inverse viewmatrix
		float const *const projmatrix;
		float const *const projmatrix_inv;
        glm::vec3 const camera_position;			// Position of the camera
		float const focal_x;						// Camera horizontal focal length
		float const focal_y;						// Camera vertical focal length
		float const tan_fovx;						// Camera horizontal field of vision
		float const tan_fovy;						// Camera vertical field of vision
        
		// textures
		Texture::TextureManager const *const d_textureManager;			// object used to retrieve specific textures by name
	};

	struct ShShaderModifiableInputs {
		// Constructor
		__device__ ShShaderModifiableInputs(PackedShShaderParams params, int idx);

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const position;				// World position of 3D Gaussian
		glm::vec3 *const scale;					// Scale of the 3D gaussian
		glm::vec4 *const rotation;				// Rotation of the 3D Gaussian
		float *const opacity;					// Opacity of the 3D gaussian
		// Contains a lof of spherical harmonics, with each degree basically defining progressively finer cubemaps. (yet still extremely coarse)
		// SHs are combined together based on constant predefined proportions and view direction.
		// -----------------------------------------------------------------
		// | SH degree | index          | multiplied by view dir component   |
		// +-----------+----------------+------------------------------------+
		// | degree 0  | index: 0       | 1             ("Base color")       |
		// | degree 1  | indexes: 1-3   | y, z, x                            |
		// | degree 2  | indexes: 4-8   | xy, yz, 2zz-xx-yy, xz, xx-yy       |
		// | degree 3  | indexes: 9-15  | (Very complex and not always used) |
		// +-----------+----------------+------------------------------------+
		glm::vec3 *const sh;

		// Precomputed 'texture' information from the neilf pbr decomposition
		glm::vec3 *const color_brdf;			// pbr splat color
		glm::vec3 *const normal;				// Splat normal in object space
		glm::vec3 *const color_base;			// Decomposed splat color without lighting
		float *const roughness;
		float *const metallic;
		glm::vec3 *const incident_light;		// total amout of light hitting this 3D Gaussian
		glm::vec3 *const local_incident_light;	// bounce light that hits this 3D gaussian
		glm::vec3 *const global_incident_light; // Global light that hits this 3D gaussian
		float *const incident_visibility;		// Fraction of how much global light hits this 3D Gaussian
	};

	struct ShShaderOutputs {
		// Constructor
		__device__ ShShaderOutputs(PackedShShaderParams params, int idx);

		float *const stencil_val; //The stencil value of the individual 3d Gaussian
		float *const stencil_opacity; // a seperate opacity for when rendering the stencil mask
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*ShShader)(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out);

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind
	std::map<std::string, int64_t> GetShShaderAddressMap();
	
	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteSHShaderCUDA(ShShader shader, int* d_splatIndexes, PackedShShaderParams packedParams);
};
