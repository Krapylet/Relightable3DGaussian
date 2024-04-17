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
        const int W; const int H;			// Sceen width and height

        // Model information:
		const int P;						// Total number of splats.
		const int splatsInShader;			// Total number of splats to be rendered with this shader.
		const int shaderStartingOffset;		// Starting index of the splats this shader needs to render.
		const float* orig_points;  			// mean 3d position of gaussian in world space.
		const float2* points_xy_image;		// mean 2d position of gaussian in screen space.

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
		const float* depths;				// Gaussian depth in view space.
		const float* colors;				// Raw Gaussian SH color.
		const float4* conic_opacity;		// ???? Float4 that contains conic something in the first 3 indexes, and opacity in the last. Read up on original splatting paper.

		// Precomputed 'texture' information from the neilf pbr decomposition
		const int S;						// Feature channel count.
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

	// Used as input and output interface to the shaders.
	// Only contains information relevant to each individual splat.
	// Acts as an interface layer that hides complexities, calculate commonly used values, thereby reducing boiler-plate code and human error.
	//TODO: Test memory and speed cost of this approach.
	//TODO: Also pass SHs? Can we do something interesting with them in the code? They function as a low-pass filter on the detail if you reduce the order.
	struct unpackedShaderParams {
		// Screen information:
        const int W; const int H;							// Sceen width and height
		// TODO: Collapse depth into a screen texture during the preprocessing (after the SH shader), so we can see the depth of the entire scene during this step.
		// This woudl be a cheap way to approximate how visible each individual splat is. 

        // splat information:
		const float orig_point;  		// mean 3d position of gaussian in world space.
		const float2 point_xy_image;		// mean 2d position of gaussian in screen space.

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
		const float depth;					// Mean splat depth in view space.
		const float color;					// Raw splat SH color. In an optimized production, this would also be used as the shader color output, but I'll keep them sperate for debugging and demonstration.
		const float4 conic_opacity;			// ???? Float4 that contains conic something in the first 3 indexes, and opacity in the last. Read up on original splatting paper.

		// Precomputed 'texture' information from the neilf pbr decomposition
		const float3 brdf_color;
		const float3 normal;
		const float3 base_color;
		const float  roughness;
		const float  metallic;
		const float  incident_light;
		const float  local_incident_light;
		const float  global_incident_light;
		const float  incident_visibility;

		// output
		// We use pointers to the output instead of return values to make it easy to extend during development.
		float* out_color;					// RGB color output the splat. Will get combined based on alpha in the next step.
	};

	//TODO: Make an unpackedShaderParams that contains direct references to a single splat's normals and such, so that the user doesn't have to worry about indexes and such.
	//The unpacking should be done in ExecuteShader()

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*shader)(shaderParams params);

	// Function pointers to the implemented shaders. Has the benefits of also being much more concise.
	__device__ extern shader outlineShader;
	__device__ extern shader wireframeShader;
	
	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(shader shader, shaderParams params);

	
};
#endif 
