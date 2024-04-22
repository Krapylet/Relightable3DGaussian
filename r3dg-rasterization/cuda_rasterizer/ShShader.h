#pragma once

#include <cuda.h>
#include "cuda_runtime.h"

#ifndef GLM_FORCE_CUDA
	#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>

namespace ShShader
{
    struct PackedShShaderParams {
        // Screen information:
        int const W; int const H;			

        // shader execution information:
		int const P;						// Total number of splats.
		int const splatsInShader;			// Total number of splats to be rendered with this shader.
		int const shaderStartingOffset;		// Starting index of the splats this shader needs to render.

    };

    struct ShShaderParams
    {
        __device__ ShShaderParams(PackedShShaderParams params, int idx);
    };

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*ShShader)(ShShaderParams params);

	// Function pointers to the implemented shaders. Has the benefits of also being much more concise.
	__device__ extern ShShader defaultShader;
	
	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(ShShader shader, PackedShShaderParams packedParams);
    
}