#include "splatShader.h"
#include "config.h"
#include <cooperative_groups.h>
#include "../utils/shaderUtils.h"
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace SplatShader
{
    __device__ SplatShaderParams::SplatShaderParams(PackedSplatShaderParams p, int idx):
        W(p.W),
        H(p.H),
        time(p.time), dt(p.dt),
		position(p.positions[idx]),

        // world space position = 
        // view space up, right, forward,
        // world space up, right, forward,  		
		screen_position(p.screen_positions[idx]),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix (p.projmatrix),
		projmatrix_inv (p.projmatrix_inv),
		focal_x (p.focal_x),
        focal_y (p.focal_y),
		tan_fovx (p.tan_fovx), 
        tan_fovy (p.tan_fovy),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),

		// pr. frame texture information
		depth (p.depths[idx]),		
		conic ((glm::vec3)p.conic_opacity[idx]), 
        color_SH (p.colors_SH + idx),

		// Precomputed 'texture' information from the neilf pbr decomposition
		color_brdf ({p.features[idx * p.S + 0], p.features[idx * p.S + 1], p.features[idx * p.S + 2]}),
		normal ({p.features[idx * p.S + 3], p.features[idx * p.S + 4], p.features[idx * p.S + 5]}),
		color_base ({p.features[idx * p.S + 6], p.features[idx * p.S + 7], p.features[idx * p.S + 8]}),
		roughness (p.features[idx * p.S + 9]),
		metallic (p.features[idx * p.S + 10]),
		incident_light (p.features[idx * p.S + 11]),
		local_incident_light (p.features[idx * p.S + 12]),
		global_incident_light (p.features[idx * p.S + 13]),
		incident_visibility (p.features[idx * p.S + 14]),
        
        // Texture information
        d_textureManager(p.d_textureManager),

        // input / output
		// can be changed, but is already populated when function is called
		opacity (((float*)p.conic_opacity) + idx * 4 + 3),  // Opacity works a bit funky because how splats are blended. It is better to multiply this paramter by something rather than setting it to specific values.

		// output
		// We use pointers to the output instead of return values to make it easy to extend during development.             
		out_color (p.out_colors + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ static void DefaultSplatShaderCUDA(SplatShaderParams p)
    {
        // Set output color
        *p.out_color = (*p.color_SH);
    }

    __device__ static void OutlineShaderCUDA(SplatShaderParams p)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = p.camera_position - p.position;
        float angle = 1 - glm::abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(p.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        // Set output color
        *p.out_color = (*p.color_SH) * opacity;
    }

    __device__ static void WireframeShaderCUDA(SplatShaderParams p)
    {
        // Get angle between splat and camera:
        glm::vec3 directionToCamera = p.camera_position - p.position;
        float angle = 1 - glm::abs(glm::dot(glm::normalize(directionToCamera), glm::normalize(p.normal)));
        // easing from https://easings.net/#easeInOutQuint
        float opacity = angle < 0.5
            ? 1 - 16 * pow(angle, 5)
            : pow(-2 * angle + 2, 5) / 2;

        float rColor = fmodf(p.time / 5000, 1.0);
        // Set output color
        *p.out_color = glm::vec3(rColor, 1 - opacity,  1 - opacity);
    }

    // Makes the object fade in and out.
    // Written in splat shader because this is where we have best access to colors.
    __device__ static void DissolveShader(SplatShaderParams p){
        cudaTextureObject_t grainyTexture = p.d_textureManager->GetTexture("Grid");

        // Grab the opacity from a mask texture
        float maskSample_xy = tex2D<float4>(grainyTexture, p.position.x, p.position.y).x;
        float maskSample_xz = tex2D<float4>(grainyTexture, p.position.x, p.position.z).x;
        float maskSample_yz = tex2D<float4>(grainyTexture, p.position.y, p.position.z).x;
        
        // combine masking from the 3 planes to create a 3d mask.
        float maskSample = maskSample_xy * maskSample_xz * maskSample_yz;

        // goes back and forth between 0 and 1 over time
        float opacityPercent = (cosf(p.time/4000) + 1)/2;

        // Offset the opacity by the mask
        float opacity = __saturatef((1 + maskSample) * opacityPercent);

        // Ease in and out of transparency with a quint easing.
        float easedOpacity = opacity < 0.5 ? 16.0 * powf(opacity, 5) : 1 - powf(-2 * opacity + 2, 5) / 2;

        float originalOpacity = *p.opacity;

        // Opacity output
        *p.opacity = easedOpacity * originalOpacity;

        // We want the colors to turn progressively more bright blue as they turn transparen
        glm::vec3 targetfadeColor = glm::vec3(0.6,0.9,1);
        float fadeColorPercent = __saturatef(1-opacity -0.3);// ;
        float fadeColorEasing = fadeColorPercent < 0.5 ? 16.0 * powf(fadeColorPercent + 0.1, 5) : 1 - powf(-2 * fadeColorPercent + 2, 5) / 2;

        // mix degree the fade color into the base color
        *p.out_color = glm::mix(*p.color_SH, targetfadeColor, fadeColorEasing);
    }

    __device__ static void CrackShaderCUDA(SplatShaderParams p)
    {
        cudaTextureObject_t crackTex = p.d_textureManager->GetTexture("Depth cracks");
        // Rescale UVs
        // Currently we just project directly downwards, but the projection can be rotated and pivoted to anywhere around the model.
        float u = p.position.x/2 - 0.5;
        float v = p.position.y/2 - 0.5;
        float crackTexDepth = 1 - tex2D<float4>(crackTex, u, v).x;

        float maxCrackDepth = 2;
        float projectionHeight = 2;
        float crackHeight = projectionHeight - crackTexDepth * maxCrackDepth;
        float splatHeight = p.position.z;

        bool crackReachesSplat = crackHeight < splatHeight;
        
        *p.opacity = crackReachesSplat ? 0 : *p.opacity;

        // Darken the areas near the cracks to create an outline.
        float CrackColorReach = 0.1f;
        float crackColorHeight = projectionHeight - (crackTexDepth + CrackColorReach) * maxCrackDepth;
        float crackColorsPercent =  1-__saturatef((splatHeight - crackColorHeight)/CrackColorReach);

        glm::vec3 hsv = RgbToHsv(*p.color_SH);
        hsv[1] *= crackColorsPercent;
        hsv[2] *= crackColorsPercent;
        *p.out_color = HsvToRgb(hsv);
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    __device__ const SplatShader defaultShader = &DefaultSplatShaderCUDA;
    __device__ const SplatShader outlineShader = &OutlineShaderCUDA;
    __device__ const SplatShader wireframeShader = &WireframeShaderCUDA;
    __device__ const SplatShader dissolveShader = &DissolveShader;
    __device__ const SplatShader crackShader = &CrackShaderCUDA;


    std::map<std::string, int64_t> GetSplatShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // there doesn't seem to be a problem casting them to int64's though.

        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(SplatShader);
        
        // Copy device shader pointers to host map
        SplatShader::SplatShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["SplatDefault"] = (int64_t)h_defaultShader;

        SplatShader::SplatShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, outlineShader, shaderMemorySize);
        shaderMap["OutlineShader"] = (int64_t)h_outlineShader;

        SplatShader::SplatShader h_wireframeShader;
        cudaMemcpyFromSymbol(&h_wireframeShader, wireframeShader, shaderMemorySize);
        shaderMap["WireframeShader"] = (int64_t)h_wireframeShader;

        SplatShader::SplatShader h_dissolveShader;
        cudaMemcpyFromSymbol(&h_dissolveShader, dissolveShader, shaderMemorySize);
        shaderMap["dissolveShader"] = (int64_t)h_dissolveShader;

        SplatShader::SplatShader h_crackShader;
        cudaMemcpyFromSymbol(&h_crackShader, crackShader, shaderMemorySize);
        shaderMap["Crack"] = (int64_t)h_crackShader;

        return shaderMap;
    }

    // ALLOCATES THE RETURN ARRAY. REMEMBER TO FREE AFTER USE.
    // Returns an array in device memory containing addresses to device shader functions.
    int64_t* GetSplatShaderAddressArray(){
        // Array is assembled on CPU before being sent to device. Addresses themselves are in device space.
        int shaderCount = 5;
        int64_t* h_shaderArray = new int64_t[shaderCount];
        size_t shaderMemorySize = sizeof(SplatShader);
 
        SplatShader::SplatShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        h_shaderArray[0] = (int64_t)h_defaultShader;

        SplatShader::SplatShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, outlineShader, shaderMemorySize);
        h_shaderArray[1] = (int64_t)h_outlineShader;

        SplatShader::SplatShader h_wireframeShader;
        cudaMemcpyFromSymbol(&h_wireframeShader, wireframeShader, shaderMemorySize);
        h_shaderArray[2] = (int64_t)h_wireframeShader;

        SplatShader::SplatShader h_dissolveShader;
        cudaMemcpyFromSymbol(&h_dissolveShader, dissolveShader, shaderMemorySize);
        h_shaderArray[3] = (int64_t)h_dissolveShader;
        
        SplatShader::SplatShader h_crackShader;
        cudaMemcpyFromSymbol(&h_crackShader, crackShader, shaderMemorySize);
        h_shaderArray[4] = (int64_t)h_crackShader;

        // copy the host array to device
        int64_t* d_shaderArray;
        cudaMalloc(&d_shaderArray, sizeof(int64_t)*shaderCount);
        cudaMemcpy(d_shaderArray, h_shaderArray, shaderMemorySize * shaderCount, cudaMemcpyDefault);

        // Delete temporary host array.
        delete[] h_shaderArray;
        return d_shaderArray;
    }


    __global__ void ExecuteShader(SplatShader* shaders, PackedSplatShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.P)
            return;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        SplatShaderParams params(packedParams, idx);

        // No need to dereference the shader function pointer.
        shaders[idx](params);
    }

}

