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

    __device__ void DefaultPostProcess(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        // We don't actuallyt do any post processing by default.
    }

    __device__ void InvertColorsShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
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
    __device__ void OutlineShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
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

    __device__ void CrackReconstructionShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        
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

    __device__ void TexturedShadows(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        // instead of darking areas with shadow by darkening the color, we instead draw a texture on top of the darnkeded areas.
        auto shadowTex = in.d_textureManager->GetTexture("shadow");
        
        float uvScale = 10; // Higher values cause the texture to repeat more often
        float u = (float)in.pixel.x / (float)in.width * uvScale;
        float v = (float)in.pixel.y / (float)in.height * uvScale;

        float lightShadow = 1 - tex2D<float4>(shadowTex, u, v).x;
        float mediumShadow = 1 - tex2D<float4>(shadowTex, u, v).z;
        float heavyShadow = 1 - tex2D<float4>(shadowTex, u, v).y;

        // Quantize light intensity
        glm::vec3 coloredLight = io.incident_light[in.pixel_idx];
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
        glm::vec3 internalColor = *io.out_shader_color * lightShadow * mediumShadow * heavyShadow;

        *io.out_shader_color = internalColor;
    }

    __device__ void ColorCorrection(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        glm::vec3 color = io.base_color[in.pixel_idx];

        // Quantize the hue of the color to simply it
        glm::vec3 hsv = RgbToHsv(color);
        hsv.r = Quantize(hsv.r, 24);
        color = HsvToRgb(hsv);

        // Reduce the shadows by 1 step in order to match the textured shadows.
        float intensity = io.incident_light[in.pixel_idx].r;
        float reducedIntensity = __saturatef(intensity + 0.25f);

        *io.out_shader_color = color * reducedIntensity;
    }

    __device__ void QuantizeLighting(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        glm::vec3 intensity = io.incident_light[in.pixel_idx];
        float whiteIntensity = max(intensity.r, max(intensity.g, intensity.b)); // Convert the light to whit efor a simpler picture.  
        float quantizedWhite = Quantize(whiteIntensity, 4);

        io.incident_light[in.pixel_idx] = glm::vec3(quantizedWhite);
    }

    __device__ void BlurLighting(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        
        // Don't blur areas that the model cover. Techinically, a better check would be writing to the stencil in an earlier shader and then comparing against that.
        glm::vec3 pix = io.incident_light[in.pixel_idx];
        bool PixelIsBackground = pix.r == 0 && pix.g == 0 && pix.b == 0;
        if(PixelIsBackground){
            return;
        } 

        io.incident_light[in.pixel_idx] = GaussianBlur((glm::vec3*)io.incident_light, in.pixel_idx, in.height, in.width);
    }


    // Apply sobel filter to depth texture in order to generate internal outlines.
    __device__ void SobelFilter(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){

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
                float depthSample = io.depth_tex[sample_pid];
                horiChange += SobelHorizontal[x+1][y+1] * depthSample * outlineStrength;
                vertChange += SobelVertical[x+1][y+1] * depthSample * outlineStrength;
            }
        }

        int depthChange = sqrt(powf(horiChange, 2) + powf(vertChange, 2));
        *io.out_shader_color = *io.out_shader_color * __saturatef(1 - abs(depthChange));
    }

    __device__ void ToonShader(PostProcessShaderInputs in, PostProcessShaderModifiableInputs io){
        ColorCorrection(in, io);
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
    __device__ const PostProcessShader blurLighting = &BlurLighting;
    __device__ const PostProcessShader quantizeLighting = &QuantizeLighting;

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
