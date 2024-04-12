/*
*  Inspired by lai Mao at https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
*
*
*
*/

#pragma once

#ifndef CUDA_SHADER_H_INCLUDED
#define CUDA_SHADER_H_INCLUDED


#include <cuda.h>
#include "cuda_runtime.h"

namespace CudaShader
{
	
	// Encapsulate shader parameters in a struct so it becomes easy to update during development.
	struct shaderParams {
		// Screen information:
        int W; int H;							// Sceen width and height

        // Gaussian information:
		int P;							// Total number of splats.
		int splatsInShader;				// Total number of splats to be rendered with this shader.
		int startingSplatIndex;			// Starting index of the splats this shader needs to render.
		const float* orig_points;  		// mean 3d position of gaussian in world space.
		float2* points_xy_image;		// mean 2d position of gaussian in screen space.

		// Projection information.
		const float* viewmatrix;
				// RightX  RightY  RightZ  0
                // UpX     UpY     UpZ     0
                // LookX   LookY   LookZ   0
                // PosX    PosY    PosZ    1
		const float* viewmatrix_inv;
				// RightX  UpX     LookX      0
                // RightY  UpY     LookY      0
                // RightZ  UpZ     LookZ      0
                // -(Pos*Right)  -(Pos*Up)  -(Pos*Look)  1
		const float* projmatrix;
		const float* projmatrix_inv;
		const float focal_x; float focal_y;
		const float tan_fovx; float tan_fovy;

		// pr. frame texture information
		float* depths;					// Gaussian depth in view space.
		float* colors;					// Raw Gaussian SH color.
		float4* conic_opacity;			// ???? Float4 that contains conic something in the first 3 indexes, and opacity in the last. Read up on original splatting paper.

		// Precomputed 'texture' information from the neilf pbr decomposition
		int S;								// Feature channel count.
		const float* features;				// Interleaved array of precomputed 'textures' for each individual gaussian. Stored in the following order:
                                            // float3 brdf_color,
                                            // float3 normal,	       Object space
                                            // float3 base_color,
                                            // float  roughness,
                                            // float  metallic
                                            // float  incident_light
                                            // float  local_incident_light
                                            // float  global_incident_light
                                            // float  incident_visibility

		// output
		float* out_color;					// Sequential RGB color output of each splat. Will get combined based on alpha in the next step.
	};

	//TODO: Make an unpackedShaderParams that contains direct references to a single splat's normals and such, so that the user doesn't have to worry about indexes and such.
	//The unpacking should be done in ExecuteShader()

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*shader)(shaderParams);

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(shader shader, shaderParams params);

	// Function pointers to the implemented shaders. Has the benefits of also being much more concise.
	extern shader outlineShader;
};
#endif 
