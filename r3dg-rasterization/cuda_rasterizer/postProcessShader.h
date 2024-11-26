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

namespace PostProcess 
{
	// Encapsulate shader parameters in a struct so it becomes easy to update during development.
	// This representation contains data for all the splats packed together.
	struct PackedPostProcessShaderParams {
        // input
		// Screen information:
        int const width; int const height;		    // render resoltion in pixels.

		// Time information
		float const time; float const dt;

		// Projection information. Probably not that usefull during post processing, but you never know.
		float const *const __restrict__ viewmatrix;
		float const *const __restrict__ viewmatrix_inv;
		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;

        // Screen texture information. Indexed with floor(screen_pos.x) + floor(screen_pos.y) * screen.width
        glm::vec3 const * background;         // Background color for the scene.
        glm::vec3 * color;          // Reglar SH derived color.
        float * opacity;        // Transparrency mask for all rendered objects in the scene.
		float * depth_tex;          // Depth texture for the scene.
		float * stencil_tex;        // Stencil texture derived form SH and splat shaders.
		glm::vec3 * surface_xyz;		// World positions at each pixel.
		glm::vec3 * pseudonormal; 		// surface normals (world space) derived from depth texture this frame
		glm::vec3 * shader_color;   // Color derived from SH and splat shader.

		// Precomputed 'texture' information from the neilf pbr decomposition
		int const  S;						            // Feature channel count.
		float * __restrict__ features;		// Screen textures stored in the following order:
											// float  roughness,
                                            // float  metallic
                                            // float  incident_visibility
                                            // float3 brdf_color,
                                            // float3 normal,	       Object space
                                            // float3 base_color,
                                            // float3  incident_light
                                            // float3  local_incident_light
                                            // float3  global_incident_light

        Texture::TextureManager *const d_textureManager;    // Object used to fetch textures uploaded by user.
	};

	// Used for double buffered swap chain
	struct PostProcessShaderBuffer {
		public:
		// only allocates memory for the struct itself. Just copies pointers.
		__host__ static PostProcessShaderBuffer CreateShallowBuffer(PackedPostProcessShaderParams params, int pixCount);

		// Allocates new memory for and deep copies the values
		__host__ static PostProcessShaderBuffer CreateDeepBuffer(PostProcessShaderBuffer srcBuffer, int pixCount);
		
		// Used to know whether to free cuda memory when the struct is destoryed.
		bool isDeepCopy;

		// Screen texture information. 
        //// Color textures:
        glm::vec3 * SH_color;          // Pure SH derived color.
        
        glm::vec3 * base_color;         // Base color of object without light. PBR decomposition derived.   
        glm::vec3 * brdf_color;         // Color of the object with lighting. PBR decomposition derived.

        //// Light textures:
        glm::vec3 * local_incident_light;   // Bounce light. PBR decomposition derived.
        glm::vec3 * global_incident_light;  // Light from the global light source. PBR decomposition derived.
        float * incident_visibility;        // Ambient occlusion. PBR decomposition derived.
        glm::vec3 * incident_light;         // The amount of lighSum of other light textures. PBR decomposition derived.

        //// Material textures:
        glm::vec3 * normal;             // Normals of surfaces in object space (same as world space when rendering a single object). PBR decomposition derived.
        float * roughness;              // roughness of objects in scene. PBR decomposition derived.
        float * metallic;               // metallicness of objects in scene. PBR decomposition derived.

        //// Scene textures:
        float * opacity;        	// Transparrency mask for all rendered objects in the scene.
		float * depth_tex;          // Depth texture for the scene.
		float * stencil_tex;        // Stencil texture. Derived form SH and splat shaders.
		glm::vec3 * surface_xyz;    // World positions at each pixel.
		glm::vec3 * pseudonormal;   // surface normals (world space) derived from depth texture this frame
        glm::vec3 * shader_color;   // Color derived from SH and splat shader. Also works as output.

		// Memory for all the feature textures are allocated to the feature buffer, so it stays contigous. It is techincally the same as the roughness pointer for now.
		// We also store scene texture memory contigously in order to reduce the numer of CUDA calls.
		float* features;
	};

	// Used as input and output interface to the shaders.
	// Only contains information relevant to each individual splat.
	// Acts as an interface layer that hides complexities and calculates commonly used values, thereby reducing boiler-plate code and human error.
	struct PostProcessShaderInputs {
		// Constructor
		__device__ PostProcessShaderInputs(PackedPostProcessShaderParams params, PostProcessShaderBuffer inputBuffer, int x, int y, int pixCount);

        // input
		// Screen information:
        int const width; int const height;		// render resoltion in pixels.
        glm::ivec2 const pixel;                        // Pixel position in screen space.
        int const pixel_idx;                          // The pixels index in the screen textures. Calculated as floor(pixel.x) + floor(pixel.y) * width.

		// Time information
		float const time; float const dt;

		// Projection information. Probably not that usefull during post processing, but you never know.
		float const *const __restrict__ viewmatrix;
		float const *const __restrict__ viewmatrix_inv;
		float const *const __restrict__ projmatrix;
		float const *const __restrict__ projmatrix_inv;
        glm::vec3 const camera_position;
		float const focal_x; float const focal_y;
		float const tan_fovx; float const tan_fovy;

		glm::vec3 const * const background;         // Background color for the scene.

        // Custom textures:
        Texture::TextureManager const *const d_textureManager;    // Object used to fetch textures uploaded by user.

		// Screen texture information. 
        //// Color textures:
        glm::vec3 * const SH_color;          // Pure SH derived color.
        glm::vec3 * const base_color;         // Base color of object without light. PBR decomposition derived.   
        glm::vec3 * const brdf_color;         // Color of the object with lighting. PBR decomposition derived.

        //// Light textures:
        glm::vec3 * const local_incident_light;   // Bounce light. PBR decomposition derived.
        glm::vec3 * const global_incident_light;  // Light from the global light source. PBR decomposition derived.
        float * const incident_visibility;        // Ambient occlusion. PBR decomposition derived.
        glm::vec3 * const incident_light;         // The amount of lighSum of other light textures. PBR decomposition derived.

        //// Material textures:
        glm::vec3 * const normal;             // Normals of surfaces in object space (same as world space when rendering a single object). PBR decomposition derived.
        float * const roughness;              // roughness of objects in scene. PBR decomposition derived.
        float * const metallic;               // metallicness of objects in scene. PBR decomposition derived.

        //// Scene textures:
        float * const opacity;        	// Transparrency mask for all rendered objects in the scene.
		float * const depth_tex;          // Depth texture for the scene.
		float * const stencil_tex;        // Stencil texture. Derived form SH and splat shaders.
		glm::vec3 * const surface_xyz;    // World positions at each pixel.
		glm::vec3 * const pseudonormal;   // surface normals (world space) derived from depth texture this frame

        glm::vec3 * const shader_color;   // Color derived from SH and splat shader. Also works as output.
	};

	// Only contains pointers to individual pixels instead of the entire textures.
	struct PostProcessShaderOutputs {
		// Copies pointers to create an instance 
		__device__ PostProcessShaderOutputs(PostProcessShaderBuffer outputBuffer, int pixelIdx);
		
		// Screen texture information. 
        //// Color textures:
        glm::vec3 * const SH_color;          // Pure SH derived color.
        glm::vec3 * const base_color;         // Base color of object without light. PBR decomposition derived.   
        glm::vec3 * const brdf_color;         // Color of the object with lighting. PBR decomposition derived.

        //// Light textures:
        glm::vec3 * const local_incident_light;   // Bounce light. PBR decomposition derived.
        glm::vec3 * const global_incident_light;  // Light from the global light source. PBR decomposition derived.
        float * const incident_visibility;        // Ambient occlusion. PBR decomposition derived.
        glm::vec3 * const incident_light;         // The amount of lighSum of other light textures. PBR decomposition derived.

        //// Material textures:
        glm::vec3 * const normal;             // Normals of surfaces in object space (same as world space when rendering a single object). PBR decomposition derived.
        float * const roughness;              // roughness of objects in scene. PBR decomposition derived.
        float * const metallic;               // metallicness of objects in scene. PBR decomposition derived.

        //// Scene textures:
        float * const opacity;        	// Transparrency mask for all rendered objects in the scene.
		float * const depth_tex;          // Depth texture for the scene.
		float * const stencil_tex;        // Stencil texture. Derived form SH and splat shaders.
		glm::vec3 * const surface_xyz;    // World positions at each pixel.
		glm::vec3 * const pseudonormal;   // surface normals (world space) derived from depth texture this frame

        glm::vec3 * const shader_color;   // Color derived from SH and splat shader. Also works as output.
	};

	// Define a shared type of fuction pointer that can point to all implemented shaders.
    typedef void (*PostProcessShader)(PostProcessShaderInputs in, PostProcessShaderOutputs out);

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind (ideally uint64_t, but pythorch only supports usigned 8-bit ints)
	std::map<std::string, int64_t> GetPostProcessShaderAddressMap();

	// Returns shader addresses in an array so they can be used in CUDA.
	int64_t* GetPostProcessShaderAddressArray();

	// Helper class for performing a deep copy
    __global__ void DeepCopy(PostProcessShaderBuffer dst, PostProcessShaderBuffer src, int pixCount);

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(PostProcessShader shader, PackedPostProcessShaderParams packedParams, PostProcessShaderBuffer inputBuffer, PostProcessShaderBuffer outputBuffer);
};
