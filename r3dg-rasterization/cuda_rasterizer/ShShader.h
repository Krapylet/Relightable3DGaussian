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

        //input
		float const time; float const dt;
		float const scale_modifier;
		dim3 const grid; 
		float const *const viewmatrix;
		float const *const viewmatrix_inv;
		float const *const projmatrix;
		float const *const projmatrix_inv;
		int const W; int const H;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;
        int deg; int max_coeffs;
		int const  S;						// Feature channel count.
		float const *const __restrict__ features;		// Interleaved array of precomputed 'textures' for each individual gaussian. Stored in the following order:
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

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const positions;
		glm::vec3 *const scales;
		glm::vec4 *const rotations;
		float *const opacities;
		glm::vec3 *const shs;

		//output
		float *const stencil_vals;
	};

	struct ShShaderParams {
		// Constructor
		__device__ ShShaderParams(PackedShShaderParams params, int idx);

        //input
		float const time; float const dt;
		float const scale_modifier;
		dim3 const grid; 
		float const *const viewmatrix;
		float const *const viewmatrix_inv;
		float const *const projmatrix;
		float const *const projmatrix_inv;
        glm::vec3 const camera_position;
		int const W; int const H;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;
        int deg; int max_coeffs;

		// Precomputed 'texture' information from the neilf pbr decomposition
		glm::vec3 const color_brdf;			// pbr splat color
		glm::vec3 const normal;				// Splat normal in object space
		glm::vec3 const color_base;			// Decomposed splat color without lighting
		float const roughness;
		float const metallic;
		glm::vec3 const incident_light;
		glm::vec3 const local_incident_light;
		glm::vec3 const global_incident_light;
		float const incident_visibility;

		Texture::TextureManager *const d_textureManager;

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const position;
		glm::vec3 *const scale;
		glm::vec4 *const rotation;
		float *const opacity;
		// Contains a lof of spherical harmonics, that are basically progressively finer cubemaps. (yet still extremely coarse)
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

		// output
		float *const stencil_val; //The stencil value of the individual splat
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*ShShader)(ShShaderParams params);

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind (ideally uint64_t, but pythorch only supports usigned 8-bit ints)
	std::map<std::string, int64_t> GetShShaderAddressMap();

	// Returns shader addresses in an array so they can be used in CUDA.
	int64_t* GetShShaderAddressArray();
	
	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(ShShader* shader, PackedShShaderParams packedParams);


	/// -------------------- Debug --------------------------
	void TestFunctionPointerMap();

	
};
