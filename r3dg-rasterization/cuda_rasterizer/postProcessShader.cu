#include "postProcessShader.h"
#include "config.h"
#include <cooperative_groups.h>
#include "../utils/shaderUtils.h"
#include "auxiliary.h"

namespace cg = cooperative_groups;

namespace PostProcess 
{
    __device__ PostProcessShaderInputs::PostProcessShaderInputs(PackedPostProcessShaderParams p, int x, int y, int pixCount):

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
           
        // Custom textures:
        d_textureManager(p.d_textureManager)    // Object used to fetch textures uploaded by user.
    {
        // Don't do anything during intialization other than setting basic values.
    }

    __device__ PostProcessShaderModifiableInputs::PostProcessShaderModifiableInputs(PackedPostProcessShaderParams p, int x, int y, int pixCount):
        SH_color(p.out_color),   

        // Feature textures: index is multiplied by number of values in all textures before them.
        roughness               (p.features + pixCount * (0)),      
        metallic                (p.features + pixCount * (1)),
        incident_visibility     (p.features + pixCount * (1+1)),
        brdf_color              ((glm::vec3 * const)(p.features + pixCount * (1+1+1))),
        normal                  ((glm::vec3 * const)(p.features + pixCount * (1+1+1+3))),    
        base_color              ((glm::vec3 * const)(p.features + pixCount * (1+1+1+3+3))),              
        incident_light          ((glm::vec3 * const)(p.features + pixCount * (1+1+1+3+3+3))),         
        local_incident_light    ((glm::vec3 * const)(p.features + pixCount * (1+1+1+3+3+3+3))),  
        global_incident_light   ((glm::vec3 * const)(p.features + pixCount * (1+1+1+3+3+3+3+3))),


        //// Scene textures:
        opacity(p.out_opacity),        // Transparrency mask for all rendered objects in the scene.
		depth_tex(p.depth_tex),          // Depth texture for the scene.
		stencil_tex(p.stencil_tex),        // Stencil texture. Derived form SH and splat shaders.
        out_surface_xyz(p.out_surface_xyz),
        out_pseudonormal(p.out_pseudonormal), 

        //TODO: Seperate post process output from SH/splat shader output, so we don't accidentally write over a result another thread needs.
        out_shader_color(&p.out_shader_color[x + y *p.width])
    {
        // Don't do anything during intialization other than setting basic values.
    }

    // Matrix that can be used for gaussian blurs
    __device__ float BlendingMatrix[] = 
    {0.009375f, 0.01875f, 0.028125f, 0.01875f, 0.009375f,
    0.01875f, 0.0375f, 0.045f, 0.0375f, 0.01875f,
    0.028125f, 0.045f, 0.3f, 0.045f, 0.028125f,
    0.01875f, 0.0375f, 0.045f, 0.0375f, 0.01875f,
    0.009375f, 0.01875f, 0.028125f, 0.01875f, 0.009375f};

    // Matrixes for sobel filter
    __device__ float SobelHorizontal[3][3] = 
    {{-1, 0, 1}, 
    {-2, 0, 2}, 
    {-1, 0, 1}};

    __device__ float SobelVertical[3][3] = 
    {{-1, -2, -1}, 
    {0, 0, 0}, 
    {1, 2, 1}};

    // helper function for converting 2D pixel cooridnates to 1D pixel IDs
    __device__ int GetPixelIdFromCoordinates(int x, int y, int screenWidth){
        return x + y * screenWidth;
    }

    // helper function for converting 1D pixel IDs to to 2D pixel coordinates
    __device__ glm::ivec2 GetPixelCoordinatesFromId(int id, int screenWidth){
        return glm::ivec2(id % screenWidth, id / screenWidth);
    }

    __device__ glm::ivec2 ClampPixelToScreen(int x, int y, int height, int width){
        int clamped_x = max(0, min(x, width));
        int clamped_y = max(0, min(y, height));
        return glm::vec2(clamped_x, clamped_y);
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

    __device__ static void DefaultPostProcess(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        // We don't actuallyt do any post processing by default.
    }

    __device__ static void InvertColorsShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        *io.out_shader_color = glm::vec3(1,1,1) - *io.out_shader_color;
    }

    __device__ bool PixelIsInsideStencil(glm::ivec2 pixel, PostProcessShaderInputs* in, PostProcessShaderModifiableInputs* io){
        // First check if the pixel is inside the screen.
        int samplePixel_idx = pixel.x + pixel.y * in->width;

        bool pixelIsOutsideScreen = samplePixel_idx < 0 || samplePixel_idx > in->height * in->width;
        if(pixelIsOutsideScreen){
            return false;
        }

        // then check the value of the stencil
        float stencilThreshold = 0.9; // Stencil values equal to or over this value will be considered "inside of stencil"
        bool pixelIsInsideStencil = io->stencil_tex[samplePixel_idx] >= stencilThreshold;

        return pixelIsInsideStencil;
    }



    // Simple method for generating an outline around the object using stencil.
    // Could be upgrade to also use depth map etc.
    __device__ static void OutlineShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        bool pixelIsOutsideOfStencil = !PixelIsInsideStencil(in.pixel, &in, &io);
        
        int outlineThickness = 5;
        int sampleDirections = 5;

        bool pixelIsNearStencil = false;

        // sample one pixel pr. thickness in all directions.
        for (float radius = 1; radius < outlineThickness + 1; radius++)
        {
            for (float direction = 0; direction <= 1; direction += 1.0f/(float)sampleDirections)
            {
                glm::ivec2 samplePixel = in.pixel + (glm::ivec2)(glm::vec2(cos(direction * 2 * M_PI), sin(direction * 2 * M_PI)) * radius);
        
                pixelIsNearStencil |= PixelIsInsideStencil(in.pixel, &in, &io);
            }
        }
        
        bool pixelShouldBeOutlined = pixelIsOutsideOfStencil && pixelIsNearStencil;

        glm::vec3 outlineColor = glm::vec3(1,0,0);

        *io.out_shader_color = io.base_color[in.pixel_idx] * (1.0f-(float)pixelShouldBeOutlined) + outlineColor * (float)pixelShouldBeOutlined;
    }

    __device__ static void CrackReconstructionShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        
        // compute mask for new surfaces
        int pid = in.pixel_idx;
        float constructionMask = io.stencil_tex[pid] * io.metallic[pid];
       
        //early exit for pixels outside of mask.
        if(constructionMask <= 0.01f){
            return;
        }

        glm::vec3 normal = io.out_pseudonormal[pid];

        // Use the slope inside the crack to apply very simple shadow to the internal color.
        // Light is shined down from above at an angle
        // TODO: Make viewDir a precalucalted value.
        glm::vec3 viewDir = glm::vec3(in.viewmatrix[9], in.viewmatrix[10], in.viewmatrix[11]); 
        glm::vec3 lightDir = glm::normalize(glm::vec3(0, -0.2f, 1));
        float lightIntensity = 0.1f;
        float ambientLight = 0.9f;
        glm::vec3 internalColor = glm::vec3(0.83f, 0.64f, 0.2f);

        // apply lighting calulation to internal color
        // We use a simple lambertian diffuse 
        internalColor *= __saturatef(__saturatef(glm::dot(lightDir, normal) * lightIntensity) + ambientLight);

        // mix the base color into the internal color near the mask edge.
        glm::vec3 outputColor = internalColor * constructionMask + *io.out_shader_color * (1-constructionMask);

        *io.out_shader_color = outputColor;
    }

    __device__ static void BlurLighting(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        /*
        *io.roughness = 0;
        for (size_t i = 0; i < 25; i++)
        {
            glm::ivec2 pixelSampleID = 
        }
        */
        
    }

    __device__ static void TexturedShadows(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        // instead of darking areas with shadow by darkening the color, we instead draw a texture on top of the darnkeded areas.
        auto shadowTex = in.d_textureManager->GetTexture("shadow");
        
        float uvScale = 20; // Higher values cause the texture to repeat more often
        float u = (float)in.pixel.x / (float)in.width * uvScale;
        float v = (float)in.pixel.y / (float)in.height * uvScale;

        float lightShadow = 1 - tex2D<float4>(shadowTex, u, v).x;
        float mediumShadow = 1 - tex2D<float4>(shadowTex, u, v).y;
        float heavyShadow = 1 - tex2D<float4>(shadowTex, u, v).z;

        // Quantize light intensity
        glm::vec3 coloredLight = io.incident_light[in.pixel_idx];
        float intensity = __max(coloredLight.x, __max(coloredLight.y, coloredLight.z));
        // all the models are very bright, so we cheat a little and make them darker
        //intensity = __powf(intensity, 2);
        intensity = roundf(intensity * 3);

        // Add progressively less light to the different types of shadow.
        heavyShadow = __saturatef(heavyShadow + intensity);
        intensity = __max(0, intensity - 0.5f);
        mediumShadow = __saturatef(mediumShadow + intensity);
        intensity = __max(0, intensity - 0.5f);
        lightShadow = __saturatef(lightShadow + intensity);        

        // Apply the shadow texture on top of the unlit colors
        glm::vec3 internalColor = *io.out_shader_color * lightShadow * mediumShadow * heavyShadow;

        *io.out_shader_color = internalColor;
    }

    __device__ static void ColorQuantization(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        glm::vec3 litColor = io.base_color[in.pixel_idx] * io.incident_light[in.pixel_idx];
        glm::vec3 internalColor = QuantizeColor(litColor, 4);

        *io.out_shader_color = internalColor;
    }

    // Apply sobel filter to depth texture in order to generate internal outlines.
    __device__ static void SobelFilter(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        
        float horiChange = 0;
        float vertChange = 0;

        for (int x = -1; x < 2; x++)
        {
            for (int y = -1; y < 2; y++)
            {
                int sample_pid = in.pixel_idx + x + y * in.width;
                float depthSample = io.depth_tex[sample_pid];
                horiChange += SobelHorizontal[x+1][y+1] * depthSample;
                vertChange += SobelVertical[x+1][y+1] * depthSample;
            }
        }

        int depthChange = sqrt(powf(horiChange, 2) + powf(vertChange, 2));
        *io.out_shader_color = *io.out_shader_color * __saturatef(1 - abs(depthChange));
    }

    __device__ static void ToonShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        ColorQuantization(in, io);
        TexturedShadows(in, io);
        SobelFilter(in, io);
    }



    __device__ const PostProcessShader defaultShader = &DefaultPostProcess;
    __device__ const PostProcessShader invertShader = &InvertColorsShader;
    __device__ const PostProcessShader outlineShader = &OutlineShader;
    __device__ const PostProcessShader crackReconstriction = &CrackReconstructionShader;
    __device__ const PostProcessShader texturedShadows = &TexturedShadows;
    __device__ const PostProcessShader sobelFilter = &SobelFilter;
    __device__ const PostProcessShader toonShader = &ToonShader;

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
        
        PostProcessShader h_texturedShadows;
        cudaMemcpyFromSymbol(&h_texturedShadows, texturedShadows, shaderMemorySize);
        shaderMap["TexturedShadows"] = (int64_t)h_texturedShadows;

        PostProcessShader h_crackReconstriction;
        cudaMemcpyFromSymbol(&h_crackReconstriction, crackReconstriction, shaderMemorySize);
        shaderMap["CrackReconstriction"] = (int64_t)h_crackReconstriction;

        PostProcessShader h_sobelFilter;
        cudaMemcpyFromSymbol(&h_sobelFilter, sobelFilter, shaderMemorySize);
        shaderMap["SobelFilter"] = (int64_t)h_sobelFilter;

        PostProcessShader h_toonShader;
        cudaMemcpyFromSymbol(&h_toonShader, toonShader, shaderMemorySize);
        shaderMap["ToonShader"] = (int64_t)h_toonShader;


        printf("Post proces address collector called\n");

        return shaderMap;
    }

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
        PostProcessShaderInputs in(packedParams, x, y, pixelCount);
        PostProcessShaderModifiableInputs io(packedParams, x, y, pixelCount);

        shader(in, io);        
    }

	
};
