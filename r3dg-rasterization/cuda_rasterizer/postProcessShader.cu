#include "postProcessShader.h"
#include "config.h"
#include <cooperative_groups.h>
#include "../utils/shaderUtils.h"
#include "auxiliary.h"

namespace cg = cooperative_groups;

namespace PostProcess 
{
    
    // Doesn't allocate new memory. Just copies pointers.
    __host__ PostProcessShaderBuffer PostProcessShaderBuffer::CreateShallowBuffer(PackedPostProcessShaderParams p, int pixCount){
        PostProcessShaderBuffer buffer;

        buffer.isDeepCopy = false;
        
        //// Feature textures
        buffer.features =               p.features ;   
        buffer.roughness =              p.features + pixCount * (0);   
        buffer.metallic =               p.features + pixCount * (1);
        buffer.incident_visibility =    p.features + pixCount * (1+1);
        buffer.brdf_color =             (glm::vec3 *)(p.features + pixCount * (1+1+1));
        buffer.normal =                 (glm::vec3 *)(p.features + pixCount * (1+1+1+3));
        buffer.base_color =             (glm::vec3 *)(p.features + pixCount * (1+1+1+3+3));      
        buffer.incident_light =         (glm::vec3 *)(p.features + pixCount * (1+1+1+3+3+3));   
        buffer.local_incident_light =   (glm::vec3 *)(p.features + pixCount * (1+1+1+3+3+3+3));  
        buffer.global_incident_light =  (glm::vec3 *)(p.features + pixCount * (1+1+1+3+3+3+3+3));

        //// Scene textures:
        buffer.opacity = p.opacity;
		buffer.depth_tex = p.depth_tex;
		buffer.stencil_tex = p.stencil_tex;
        buffer.surface_xyz = p.surface_xyz;
        buffer.pseudonormal = p.pseudonormal; 
        buffer.shader_color = p.shader_color;
        buffer.SH_color = p.color;

        return buffer;
   }

    
    // Helper class for performing a deep copy
    __global__ void DeepCopy(PostProcessShaderBuffer dst, PostProcessShaderBuffer src, int pixCount){
        auto idx = cg::this_grid().thread_rank();
        if (idx >= pixCount)
            return;

        // copy feature textures
        dst.roughness[idx] = src.roughness[idx];
        dst.metallic[idx] = src.metallic[idx];
        dst.incident_visibility[idx] = src.incident_visibility[idx];
        dst.brdf_color[idx] = src.brdf_color[idx];
        dst.normal[idx] = src.normal[idx];
        dst.base_color[idx] = src.base_color[idx];                     
        dst.incident_light[idx] = src.incident_light[idx];            
        dst.local_incident_light[idx] = src.local_incident_light[idx]; 
        dst.global_incident_light[idx] = src.global_incident_light[idx];

        // copy Scene textures:
        dst.opacity[idx] = src.opacity[idx];
        dst.depth_tex[idx] = src.depth_tex[idx]; 
		dst.stencil_tex[idx] = src.stencil_tex[idx];
        dst.surface_xyz[idx] = src.surface_xyz[idx];         
        dst.pseudonormal[idx] = src.pseudonormal[idx];        
        dst.shader_color[idx] = src.shader_color[idx];
        dst.SH_color[idx] = src.SH_color[idx];
    }


    // perform a deep copy of the input parameters, but allocate new memory
    __host__ PostProcessShaderBuffer PostProcessShaderBuffer::CreateDeepBuffer(PostProcessShaderBuffer srcBuffer, int pixCount)
    {
        PostProcessShaderBuffer buffer;

        buffer.isDeepCopy = true;

        ////  allocate new memory to textures. We allocate it all in one go to reduce overhead.
        // Feature textures
        int featureMemSize = 21;
        int sceneMemSize = 15;
        checkCudaErrors(cudaMalloc(&buffer.features, pixCount * (featureMemSize + sceneMemSize) * sizeof(float)));
        float* bufferStart = buffer.features;
        buffer.roughness =                bufferStart + pixCount * (0);     
        buffer.metallic =                 bufferStart + pixCount * (1);
        buffer.incident_visibility =      bufferStart + pixCount * (1+1);
        buffer.brdf_color =               (glm::vec3 *)(bufferStart + pixCount * (1+1+1));
        buffer.normal =                   (glm::vec3 *)(bufferStart + pixCount * (1+1+1+3));    
        buffer.base_color =               (glm::vec3 *)(bufferStart + pixCount * (1+1+1+3+3));                    
        buffer.incident_light =           (glm::vec3 *)(bufferStart + pixCount * (1+1+1+3+3+3));            
        buffer.local_incident_light =     (glm::vec3 *)(bufferStart + pixCount * (1+1+1+3+3+3+3));   
        buffer.global_incident_light =    (glm::vec3 *)(bufferStart + pixCount * (1+1+1+3+3+3+3+3));

        // Scene textures:
        buffer.opacity =                bufferStart + pixCount * (featureMemSize);  
		buffer.depth_tex =              bufferStart + pixCount * (featureMemSize+1); 
		buffer.stencil_tex =            bufferStart + pixCount * (featureMemSize+1+1); 
        buffer.surface_xyz =            (glm::vec3*)( bufferStart + pixCount * (featureMemSize+1+1+1)); 
        buffer.pseudonormal =           (glm::vec3*)( bufferStart + pixCount * (featureMemSize+1+1+1+3)); 
        buffer.shader_color =           (glm::vec3*)( bufferStart + pixCount * (featureMemSize+1+1+1+3+3)); 
        buffer.SH_color =               (glm::vec3*)( bufferStart + pixCount * (featureMemSize+1+1+1+3+3+3)); 

        //// Perfrom deep copy
        DeepCopy<<<(pixCount + 255) / 256, 256>>>(buffer, srcBuffer, pixCount);
        CHECK_CUDA(, true);
        return buffer;
    }

    __device__ PostProcessShaderInputs::PostProcessShaderInputs(PackedPostProcessShaderParams p, PostProcessShaderBuffer in, int x, int y, int pixCount):

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
        d_textureManager(p.d_textureManager),    // Object used to fetch textures uploaded by user.

        // Feature textures: index is multiplied by number of values in all textures before them.
        roughness               (in.roughness),      
        metallic                (in.metallic),
        incident_visibility     (in.incident_visibility),
        brdf_color              (in.brdf_color),
        normal                  (in.normal),
        base_color              (in.base_color),           
        incident_light          (in.incident_light),    
        local_incident_light    (in.local_incident_light),  
        global_incident_light   (in.global_incident_light),


        //// Scene textures:
        opacity(in.opacity),        // Transparrency mask for all rendered objects in the scene.
		depth_tex(in.depth_tex),          // Depth texture for the scene.
		stencil_tex(in.stencil_tex),        // Stencil texture. Derived form SH and splat shaders.
        surface_xyz(in.surface_xyz),
        pseudonormal(in.pseudonormal), 
        shader_color(in.shader_color),
        SH_color(in.SH_color)
    {
        // Don't do anything during intialization other than setting basic values.
    }

    __device__ PostProcessShaderOutputs::PostProcessShaderOutputs(PostProcessShaderBuffer in, int pixelIdx):
        // Feature textures:
        roughness               (&in.roughness[pixelIdx]),      
        metallic                (&in.metallic[pixelIdx]),
        incident_visibility     (&in.incident_visibility[pixelIdx]),
        brdf_color              (&in.brdf_color[pixelIdx]),
        normal                  (&in.normal[pixelIdx]),
        base_color              (&in.base_color[pixelIdx]),           
        incident_light          (&in.incident_light[pixelIdx]),    
        local_incident_light    (&in.local_incident_light[pixelIdx]),  
        global_incident_light   (&in.global_incident_light[pixelIdx]),

        // Scene textures:
        opacity(&in.opacity[pixelIdx]),        
		depth_tex(&in.depth_tex[pixelIdx]),          
		stencil_tex(&in.stencil_tex[pixelIdx]),        
        surface_xyz(&in.surface_xyz[pixelIdx]),
        pseudonormal(&in.pseudonormal[pixelIdx]), 
        shader_color(&in.shader_color[pixelIdx]),
        SH_color(&in.SH_color[pixelIdx])
    {
        // Don't do anything during intialization other than setting basic values.
    }

    __device__ void DefaultPostProcess(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        // We don't actuallyt do any post processing by default.
    }

    __device__ void InvertColorsShader(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        *out.shader_color = glm::vec3(1,1,1) - in.shader_color[in.pixel_idx];
    }

    __device__ bool PixelIsInsideStencil(glm::ivec2 pixel, PostProcessShaderInputs* in){
        // First check if the pixel is inside the screen.
        int samplePixel_idx = pixel.x + pixel.y * in->width;

        bool pixelIsOutsideScreen = samplePixel_idx < 0 || samplePixel_idx > in->height * in->width;
        if(pixelIsOutsideScreen){
            return false;
        }

        // then check the value of the stencil
        float stencilThreshold = 0.9; // Stencil values equal to or over this value will be considered "inside of stencil"
        bool pixelIsInsideStencil = in->stencil_tex[samplePixel_idx] >= stencilThreshold;

        return pixelIsInsideStencil;
    }



    // Simple method for generating an outline around the object using stencil.
    // Could be upgrade to also use depth map etc.
    __device__ void OutlineShader(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        bool pixelIsOutsideOfStencil = !PixelIsInsideStencil(in.pixel, &in);
        
        int outlineThickness = 5;
        int sampleDirections = 5;

        bool pixelIsNearStencil = false;

        // sample one pixel pr. thickness in all directions.
        for (float radius = 1; radius < outlineThickness + 1; radius++)
        {
            for (float direction = 0; direction <= 1; direction += 1.0f/(float)sampleDirections)
            {
                glm::ivec2 samplePixel = in.pixel + (glm::ivec2)(glm::vec2(cos(direction * 2 * M_PI), sin(direction * 2 * M_PI)) * radius);
        
                pixelIsNearStencil |= PixelIsInsideStencil(in.pixel, &in);
            }
        }
        
        bool pixelShouldBeOutlined = pixelIsOutsideOfStencil && pixelIsNearStencil;

        glm::vec3 outlineColor = glm::vec3(1,0,0);

        *out.shader_color = in.base_color[in.pixel_idx] * (1.0f-(float)pixelShouldBeOutlined) + outlineColor * (float)pixelShouldBeOutlined;
    }

    __device__ void CrackReconstructionShader(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        
        // compute mask for new surfaces
        int pid = in.pixel_idx;
        float constructionMask = in.stencil_tex[pid] * in.metallic[pid];
       
        //early exit for pixels outside of mask.
        if(constructionMask <= 0.01f){
            return;
        }

        glm::vec3 normal = in.pseudonormal[pid];

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
        glm::vec3 outputColor = internalColor * constructionMask + in.shader_color[in.pixel_idx] * (1-constructionMask);

        *out.shader_color = outputColor;
    }

    __device__ void TexturedShadows(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        // exit early if not inside model stencil
        if(in.stencil_tex[in.pixel_idx] < 0.01f)
        {
            *out.shader_color = glm::vec3(1);
            return;
        }

        // instead of darking areas with shadow by darkening the color, we instead draw a texture on top of the darnkeded areas.
        auto shadowTex = in.d_textureManager->GetTexture("shadow");
        
        float uvScale = 10; // Higher values cause the texture to repeat more often
        float u = (float)in.pixel.x / (float)in.width * uvScale;
        float v = (float)in.pixel.y / (float)in.height * uvScale;

        float lightShadow = 1 - tex2D<float4>(shadowTex, u, v).x;
        float mediumShadow = 1 - tex2D<float4>(shadowTex, u, v).z;
        float heavyShadow = 1 - tex2D<float4>(shadowTex, u, v).y;

        // Quantize light intensity
        glm::vec3 coloredLight = in.incident_light[in.pixel_idx];
        float intensity = __max(coloredLight.x, __max(coloredLight.y, coloredLight.z));
        intensity = roundf(intensity * 4);

        // Add progressively less light to the different types of shadow.
        // We multiply intensity by 1 extra in order to reduce the step at which the shadows become visible.
        heavyShadow = __saturatef(heavyShadow + intensity);
        intensity = __max(0, intensity - 1.0f);
        mediumShadow = __saturatef(mediumShadow + intensity);
        intensity = __max(0, intensity - 1.0f);
        lightShadow = __saturatef(lightShadow + intensity);        

        // Apply the shadow texture on top of the unlit colors
        glm::vec3 internalColor = *out.shader_color * lightShadow * mediumShadow * heavyShadow;

        *out.shader_color = internalColor;
    }

    __device__ void ColorCorrection(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        glm::vec3 color = in.base_color[in.pixel_idx];

        // Quantize the hue of the color to simply it
        glm::vec3 hsv = RgbToHsv(color);
        hsv.r = Quantize(hsv.r, 24);
        color = HsvToRgb(hsv);

        // Reduce the shadows by 1 step in order to match the textured shadows.
        float intensity = in.incident_light[in.pixel_idx].r;
        float reducedIntensity = __saturatef(intensity + 0.25f);

        *out.shader_color = color * reducedIntensity;
    }

    __device__ void QuantizeLighting(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        glm::vec3 intensity = in.incident_light[in.pixel_idx];
        float whiteIntensity = max(intensity.r, max(intensity.g, intensity.b)); // Convert the light to whit efor a simpler picture.  
        float quantizedWhite = Quantize(whiteIntensity, 4);

        *out.incident_light = glm::vec3(quantizedWhite);
    }

    __device__ void BlurLighting(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        
        // Don't blur areas that the model cover. Techinically, a better check would be writing to the stencil in an earlier shader and then comparing against that.
        glm::vec3 pix = in.incident_light[in.pixel_idx];
        bool PixelIsBackground = pix.r == 0 && pix.g == 0 && pix.b == 0;
        if(PixelIsBackground){
            return;
        } 

        *out.incident_light = GaussianBlur(in.incident_light, in.pixel_idx, in.height, in.width);
    }


    // Apply sobel filter to depth texture in order to generate internal outlines.
    __device__ void SobelFilter(PostProcessShaderInputs in, PostProcessShaderOutputs out){

        // Matrixes for sobel filter
        float SobelHorizontal[3][3] = 
        {{-1, 0, 1}, 
        {-2, 0, 2}, 
        {-1, 0, 1}};

        float SobelVertical[3][3] = 
        {{-1, -2, -1}, 
        {0, 0, 0}, 
        {1, 2, 1}};
        
        float outlineStrength = 2;

        float horiChange = 0;
        float vertChange = 0;

        for (int x = -1; x < 2; x++)
        {
            for (int y = -1; y < 2; y++)
            {
                int sample_pid = in.pixel_idx + x + y * in.width;
                float depthSample = in.depth_tex[sample_pid];
                horiChange += SobelHorizontal[x+1][y+1] * depthSample * outlineStrength;
                vertChange += SobelVertical[x+1][y+1] * depthSample * outlineStrength;
            }
        }

        int depthChange = sqrt(powf(horiChange, 2) + powf(vertChange, 2));
        *out.shader_color *= __saturatef(1 - abs(depthChange));
    }

    __device__ void ToonShader(PostProcessShaderInputs in, PostProcessShaderOutputs out){
        ColorCorrection(in, out);
        TexturedShadows(in, out);
        SobelFilter(in, out);
    }

    __device__ PostProcessShader defaultShader = &DefaultPostProcess;
    __device__ PostProcessShader invertShader = &InvertColorsShader;
    __device__ PostProcessShader outlineShader = &OutlineShader;
    __device__ PostProcessShader crackReconstriction = &CrackReconstructionShader;
    __device__ PostProcessShader texturedShadows = &TexturedShadows;
    __device__ PostProcessShader sobelFilter = &SobelFilter;
    __device__ PostProcessShader toonShader = &ToonShader;
    __device__ PostProcessShader blurLighting = &BlurLighting;
    __device__ PostProcessShader quantizeLighting = &QuantizeLighting;

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

        PostProcessShader h_blurLighting;
        cudaMemcpyFromSymbol(&h_blurLighting, blurLighting, shaderMemorySize);
        shaderMap["BlurLighting"] = (int64_t)h_blurLighting;

        PostProcessShader h_quantizeLighting;
        cudaMemcpyFromSymbol(&h_quantizeLighting, quantizeLighting, shaderMemorySize);
        shaderMap["QuantizeLighting"] = (int64_t)h_quantizeLighting;


        printf("Post proces address collector called\n");

        return shaderMap;
    }


	// Executes a shader on the GPU with the given parameters.
	__global__ extern void ExecuteShader(PostProcessShader shader, PackedPostProcessShaderParams packedParams, PostProcessShaderBuffer inputBuffer, PostProcessShaderBuffer outputBuffer){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        int pixelCount = packedParams.width * packedParams.height;
        if (idx >= pixelCount)
            return;

        int x = idx % packedParams.width;
        int y = idx / packedParams.height; // We use int divison to discard remainder

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        PostProcessShaderInputs in(packedParams, inputBuffer, x, y, pixelCount);
        PostProcessShaderOutputs out(outputBuffer, x + y * packedParams.width);

        shader(in, out);     
    }
};
