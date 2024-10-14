#include "postProcessShader.h"
#include "config.h"
#include <cooperative_groups.h>
#include "../utils/shaderUtils.h"
#include "auxiliary.h"

namespace cg = cooperative_groups;

namespace PostProcess 
{
    __device__ PostProcessShaderParams::PostProcessShaderParams(PackedPostProcessShaderParams p, int x, int y, int pixCount):

        width(p.width), height(p.height),		
        pixel(glm::vec2(x, y)),               
        pixel_idx(x + y *p.width),               

		// Time information
		time(p.time),
        dt(p.dt),

		// Projection information. Probably not that usefull during post processing, but you never know.
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix(p.projmatrix),
		projmatrix_inv(p.projmatrix_inv),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),
		focal_x(p.focal_x), focal_y(p.focal_y),
		tan_fovx(p.tan_fovx), tan_fovy(p.tan_fovy),

        background(p.background),              
        SH_color(p.out_color),                

        // Feature textures: index is multiplied by number of values in all textures before them.
        brdf_color              ((glm::vec3 const * const)p.features + pixCount * (0)),
        normal                  ((glm::vec3 const * const)p.features + pixCount * (3)),    
        base_color              ((glm::vec3 const * const)p.features + pixCount * (3+3)),              
        roughness               (p.features + pixCount * (3+3+3)),      
        metallic                (p.features + pixCount * (3+3+3+1)),
        incident_light          ((glm::vec3 const * const)p.features + pixCount * (3+3+3+1+1)),         
        local_incident_light    ((glm::vec3 const * const)p.features + pixCount * (3+3+3+1+1+3)),  
        global_incident_light   ((glm::vec3 const * const)p.features + pixCount * (3+3+3+1+1+3+3)),
        incident_visibility     (p.features + pixCount * (3+3+3+1+1+3+3+3)),

        //// Scene textures:
        opacity(p.out_opacity),        // Transparrency mask for all rendered objects in the scene.
		depth_tex(p.depth_tex),          // Depth texture for the scene.
		stencil_tex(p.stencil_tex),        // Stencil texture. Derived form SH and splat shaders.

        // Custom textures:
        d_textureManager(p.d_textureManager),    // Object used to fetch textures uploaded by user.

        // input / output
        out_shader_color(&p.out_shader_color[pixel_idx])
    {
        // Don't do anything during intialization other than setting basic values.
    }

    __device__ static void DefaultPostProcess(PostProcessShaderParams p){
        // We don't actuallyt do any post processing by default.
    }

    __device__ static void InvertColorsShader(PostProcessShaderParams p){
        *p.out_shader_color = glm::vec3(1,1,1) - *p.out_shader_color;
    }

    __device__ const PostProcessShader defaultShader = &DefaultPostProcess;
    __device__ const PostProcessShader invertShader = &InvertColorsShader;

	// Returns a map of shader names and shader device function pointers that can be passed back to the python frontend though pybind.
	// we cast pointers to int since pure pointers aren't supported by pybind (ideally uint64_t, but pythorch only supports usigned 8-bit ints)
	//std::map<std::string, int64_t> GetPostProcessShaderAddressMap();

	// Returns shader addresses in an array so they can be used in CUDA.
	//int64_t* GetPostProcessShaderAddressArray();

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(PackedPostProcessShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        int pixelCount = packedParams.width * packedParams.height;
        if (idx >= pixelCount)
            return;
        
        int x = idx % packedParams.width;
        int y = idx / packedParams.height; // We use int divison to discard remainder

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        PostProcessShaderParams params(packedParams, x, y, pixelCount);

        // Debug print statement for seeing what's going on inside shader kernels.
        //if (idx == 1)
            //printf("Post process running for pixel (%i, %i)\n", x, y);

        // No need to dereference the shader function pointer.
        //shader(params);
        invertShader(params);
    }

	
};
