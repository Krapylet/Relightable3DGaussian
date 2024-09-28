#include "splatShader.h"
#include "config.h"
#include <cooperative_groups.h>


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
		conic_opacity (p.conic_opacity[idx]), // Todo: split opacity to own variable
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

    __device__ static void TextureTestShaderCUDA(SplatShaderParams p)
    {

        char* texName = "Cracks";
        cudaTextureObject_t crackTex = p.d_textureManager->GetTexture(texName);

        float4 sampleColor = tex2D<float4>(crackTex, p.position.x, p.position.y);
        
        *p.out_color = glm::vec3(sampleColor.x, sampleColor.y, sampleColor.z);

        //TODO: Make opacity something that can be modified in the shader.
        //p.conic_opacity.w = 1;
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    __device__ const SplatShader defaultShader = &DefaultSplatShaderCUDA;
    __device__ const SplatShader outlineShader = &OutlineShaderCUDA;
    __device__ const SplatShader wireframeShader = &WireframeShaderCUDA;
    __device__ const SplatShader textureTestShader = &TextureTestShaderCUDA;


    IndirectMap<char*, SplatShader>* GetSplatShaderAddressMap(){
        std::vector<char*> shaderNames;
        std::vector<SplatShader> shaderFunctionPointers;
        size_t shaderMemorySize = sizeof(SplatShader);
        
        // Copy device shader pointers to host, so we can store them in the indirect map.
        SplatShader::SplatShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderFunctionPointers.push_back(h_defaultShader);
        shaderNames.push_back("SplatDefault");

        SplatShader::SplatShader h_outlineShader;
        cudaMemcpyFromSymbol(&h_outlineShader, outlineShader, shaderMemorySize);
        shaderFunctionPointers.push_back(h_outlineShader);
        shaderNames.push_back("OutlineShader");

        SplatShader::SplatShader h_wireframeShader;
        cudaMemcpyFromSymbol(&h_wireframeShader, wireframeShader, shaderMemorySize);
        shaderFunctionPointers.push_back(h_wireframeShader);
        shaderNames.push_back("WireframeShader");

        SplatShader::SplatShader h_textureTestShader;
        cudaMemcpyFromSymbol(&h_textureTestShader, textureTestShader, shaderMemorySize);
        shaderFunctionPointers.push_back(h_textureTestShader);
        shaderNames.push_back("TextureTestShader");

        IndirectMap<char*, SplatShader>* splatShaderMap = new IndirectMap<char*, SplatShader>(shaderNames, shaderFunctionPointers);

        return splatShaderMap;
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

