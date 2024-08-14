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

#ifndef GLM_FORCE_CUDA
	#define GLM_FORCE_CUDA
#endif

#include <glm/glm.hpp>

namespace ShShader
{
	struct PackedShShaderParams {
        const int P;

        //input
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

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const positions;
		glm::vec3 *const scales;
		glm::vec4 *const rotations;
		float *const opacities;
		float *const shs;
	};

	struct ShShaderParams {
		// Constructor
		__device__ ShShaderParams(PackedShShaderParams params, int idx);

        //input
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

		//input/output   -   contains values when the method is called that can be changed.
		glm::vec3 *const position;
		glm::vec3 *const scale;
		glm::vec4 *const rotation;
		float *const opacity;
		float *const sh;
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*ShShader)(ShShaderParams params);

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind (ideally uint64_t, but pythorch only supports usigned 8-bit ints)
	std::map<std::string, int64_t> GetShShaderAddressMap();
	
	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(ShShader*, PackedShShaderParams packedParams);

	
};
