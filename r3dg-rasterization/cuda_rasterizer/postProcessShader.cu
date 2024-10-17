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
        pixel(x, y),               
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
        roughness               (p.features + pixCount * (0)),      
        metallic                (p.features + pixCount * (1)),
        incident_visibility     (p.features + pixCount * (1+1)),
        brdf_color              ((glm::vec3 const * const)(p.features + pixCount * (1+1+1))),
        normal                  ((glm::vec3 const * const)(p.features + pixCount * (1+1+1+3))),    
        base_color              ((glm::vec3 const * const)(p.features + pixCount * (1+1+1+3+3))),              
        incident_light          ((glm::vec3 const * const)(p.features + pixCount * (1+1+1+3+3+3))),         
        local_incident_light    ((glm::vec3 const * const)(p.features + pixCount * (1+1+1+3+3+3+3))),  
        global_incident_light   ((glm::vec3 const * const)(p.features + pixCount * (1+1+1+3+3+3+3+3))),


        //// Scene textures:
        opacity(p.out_opacity),        // Transparrency mask for all rendered objects in the scene.
		depth_tex(p.depth_tex),          // Depth texture for the scene.
		stencil_tex(p.stencil_tex),        // Stencil texture. Derived form SH and splat shaders.

        // Custom textures:
        d_textureManager(p.d_textureManager),    // Object used to fetch textures uploaded by user.

        // input / output
        //TODO: Seperate post process output from SH/splat shader output, so we don't accidentally write over a result another thread needs.
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

    __device__ bool PixelIsInsideStencil(glm::ivec2 pixel, PostProcessShaderParams* p){
        // First check if the pixel is inside the screen.
        int samplePixel_idx = pixel.x + pixel.y * p->width;

        bool pixelIsOutsideScreen = samplePixel_idx < 0 || samplePixel_idx > p->height * p->width;
        if(pixelIsOutsideScreen){
            return false;
        }

        // then check the value of the stencil
        float stencilThreshold = 0.9; // Stencil values equal to or over this value will be considered "inside of stencil"
        bool pixelIsInsideStencil = p->stencil_tex[samplePixel_idx] >= stencilThreshold;

        return pixelIsInsideStencil;
    }

    // Performs a very simple quantization of the model colors
    __device__ static glm::vec3 QuantizeColor(glm::vec3 color, int steps)
    {
        // for each component of the color, clamp it to the closest multiple of the step threshold (1/steps).
        float quantizedR = roundf(color.r * steps)/steps;
        float quantizedG = roundf(color.g * steps)/steps;
        float quantizedB = roundf(color.b * steps)/steps;

        glm::vec3 quatnizedColor(quantizedR, quantizedG, quantizedB);
        return quatnizedColor;
    }

    // Simple method for generating an outline around the object using stencil.
    __device__ static void OutlineShader(PostProcessShaderParams p){
        bool pixelIsOutsideOfStencil = !PixelIsInsideStencil(p.pixel, &p);
        
        int outlineThickness = 5;
        int sampleDirections = 5;

        bool pixelIsNearStencil = false;

        // sample one pixel pr. thickness in all directions.
        for (float radius = 1; radius < outlineThickness + 1; radius++)
        {
            for (float direction = 0; direction <= 1; direction += 1.0f/(float)sampleDirections)
            {
                glm::ivec2 samplePixel = p.pixel + (glm::ivec2)(glm::vec2(cos(direction * 2 * M_PI), sin(direction * 2 * M_PI)) * radius);
        
                pixelIsNearStencil |= PixelIsInsideStencil(samplePixel, &p);
            }
        }
        
        bool pixelShouldBeOutlined = pixelIsOutsideOfStencil && pixelIsNearStencil;

        glm::vec3 outlineColor = glm::vec3(1,0,0);

        glm::vec3 litColor = p.base_color[p.pixel_idx] * p.incident_light[p.pixel_idx];
        glm::vec3 internalColor = QuantizeColor(litColor, 4);

        *p.out_shader_color = internalColor * (1.0f-(float)pixelShouldBeOutlined) + outlineColor * (float)pixelShouldBeOutlined;
        //*p.out_shader_color = glm::vec3(p.roughness[p.pixel_idx],p.roughness[p.pixel_idx],p.roughness[p.pixel_idx]);
    }



    __device__ const PostProcessShader defaultShader = &DefaultPostProcess;
    __device__ const PostProcessShader invertShader = &InvertColorsShader;
    __device__ const PostProcessShader outlineShader = &OutlineShader;

	std::map<std::string, int64_t> GetPostProcessShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // there doesn't seem to be a problem casting them to int64's though.

        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(PostProcessShader);
        
        // Copy device shader pointers to host map
        PostProcessShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["SplatDefault"] = (int64_t)h_defaultShader;

        PostProcessShader h_invertShader;
        cudaMemcpyFromSymbol(&h_invertShader, invertShader, shaderMemorySize);
        shaderMap["Invert"] = (int64_t)h_invertShader;

        PostProcessShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, outlineShader, shaderMemorySize);
        shaderMap["Outline"] = (int64_t)h_outlineShader;


        printf("Post proces address collector called\n");

        return shaderMap;
    }

	// Returns shader addresses in an array so they can be used in CUDA.
	//int64_t* GetPostProcessShaderAddressArray();

	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(PostProcessShader shader, PackedPostProcessShaderParams packedParams){
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
        /*
        if (idx == 80050){
            // First check if the pixel is inside the screen.
            int pixel_idx = x + y * params.width;

            bool pixelIsOutsideScreen = pixel_idx < 0 || pixel_idx > params.height * params.width;

            float stencilThreshold = 1; // Stencil values equal to or over this value will be considered "inside of stencil"
            bool pixelIsInsideStencil = params.stencil_tex[p->pixel_idx] >= stencilThreshold;
        }
        */

        shader(params);        
    }

	
};
