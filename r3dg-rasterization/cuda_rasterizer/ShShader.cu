#include "shShader.h"
#include <iostream>
#include <cooperative_groups.h>
#include "../utils/texture.h"
#include <math.h>
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace ShShader
{
    __device__ ShShaderConstantInputs::ShShaderConstantInputs(PackedShShaderParams p, int idx):
        time(p.time), dt(p.dt),
        scale_modifier(p.scale_modifier),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix(p.projmatrix),
		projmatrix_inv(p.projmatrix_inv),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),
		W(p.W), H(p.H),
		focal_x(p.focal_x), focal_y(p.focal_y),
		tan_fovx(p.tan_fovx), tan_fovy(tan_fovy),
        deg(p.deg), max_coeffs(p.max_coeffs),

        d_textureManager(p.d_textureManager)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ ShShaderModifiableInputs::ShShaderModifiableInputs(PackedShShaderParams p, int idx):
        // Precomputed pr. splat 'texture' information from the neilf pbr decomposition
        roughness (p.features + idx * p.S + 0),
		metallic (p.features + idx * p.S + 1),
		incident_visibility (p.features + idx * p.S + 2),
		color_brdf ((glm::vec3*)&p.features[idx * p.S + 3]),
		normal ((glm::vec3*)&p.features[idx * p.S + 6]),
		color_base ((glm::vec3*)&p.features[idx * p.S + 9]),
		incident_light ((glm::vec3*)&p.features[idx * p.S + 12]),
		local_incident_light ((glm::vec3*)&p.features[idx * p.S + 15]),
		global_incident_light ((glm::vec3*)&p.features[idx * p.S + 18]),

		position(p.positions + idx),
		scale(p.scales + idx),
		rotation(p.rotations + idx),
		opacity(p.opacities + idx),
		sh(p.shs + idx * p.max_coeffs) // could also be calculated as idx + (p.deg + 1)^2
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ ShShaderOutputs::ShShaderOutputs(PackedShShaderParams p, int idx):
        stencil_val(p.stencil_vals + idx),
        stencil_opacity(p.stencil_opacities + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ void DefaultShShaderCUDA(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out)
    {
        // Default shader doesn't need to set any values.
    }

    __device__ void ExponentialPositionShaderCUDA(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out)
    {
        // multiply sh position and scale by y coordinate
        float posX = abs((*io.position).x);
        float posY = abs((*io.position).y);
        float posZ = abs((*io.position).z);
        float dist = glm::length(*io.position);
        
        *io.scale = glm::vec3((*io.scale).x * posY, (*io.scale).y * 2, (*io.scale).z) * posY;
        *io.position = glm::vec3((*io.position).x * posY, (*io.position).y * 2, (*io.position).z) * posY;
    }

    // A shader that makes an object pulse with a heartbeat
    // The heart has two concurrent beats that we model here.
    // Written as SH shader because this is where we can update positions and scales.
    __device__ void HeartbeatShaderCUDA(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out)
    {
        // Sample the texture contiaing the pattern for growth with the atreal pulse.
        cudaTextureObject_t atrialTex = in.d_textureManager->GetTexture("Turbulence");
        float atrialSampleXY = tex2D<float4>(atrialTex, io.position->x, io.position->y).x;
        float atrialSampleXZ = tex2D<float4>(atrialTex, io.position->x, io.position->z).x;
        float atrialSampleYZ = tex2D<float4>(atrialTex, io.position->y, io.position->z).x;
        float atrialPattern = (atrialSampleXY + atrialSampleXZ + atrialSampleYZ) / 3;

        // Sample for ventricular pattern
        cudaTextureObject_t ventricularTex = in.d_textureManager->GetTexture("Craters");
        float ventricularSampleXY = 1-tex2D<float4>(ventricularTex, io.position->x, io.position->y).x;
        float ventricularSampleXZ = 1-tex2D<float4>(ventricularTex, io.position->x, io.position->z).x;
        float ventricularSampleYZ = 1-tex2D<float4>(ventricularTex, io.position->y, io.position->z).x;
        float ventricularPattern = (ventricularSampleXY + ventricularSampleXZ + ventricularSampleYZ) / 3;
 
        // Create a wave that pulses out from the center of the object with values from 0-1.
        // Technically, this should instead expand from the center of the object, but since the objects don't currently have a pivot when
        // when only rendinering a single object, I'm using the origin of the world instead.
        float pulsePeriod = 1;
        float distInfluence = -0.5;
        float splatDist = glm::length(*io.position);
        float time = in.time/1000/pulsePeriod + splatDist * distInfluence;

        // Function i came up with myself for approximating the volume of a heartbeat with a regular rythm.
        // 1/4th conrraction and 3/4th expansion. Is graphed here: https://graphtoy.com/?f1(x,t)=cos(x)&v1=false&f2(x,t)=cos(x*3)&v2=false&f3(x,t)=1-round(sin(x)/2+0.5)&v3=false&f4(x,t)=f2(x)*f3(x)&v4=false&f5(x,t)=round(sin(x/4*3)/2+0.5)&v5=false&f6(x,t)=(f1(x%25(%F0%9D%9C%8B*4/3))*(1-f3(x%25(%F0%9D%9C%8B*4/3)))+f4(x%25(%F0%9D%9C%8B*4/3))+1)/2&v6=true&grid=1&coords=3.966845761855698,-0.05408620270927578,6.1301195710745375
        // https://graphtoy.com/?f1(x,t)=%F0%9D%9C%8B*4/3&v1=false&f2(x,t)=&v2=false&f3(x,t)=&v3=false&f4(x,t)=&v4=true&f5(x,t)=&v5=true&f6(x,t)=(cos(x%25f1())*round(sin(x%25f1())/2+0.5)+cos%20((x%25f1())*3)*(1-round(sin(x%25f1())/2+0.5))%20+%201)/2&v6=true&grid=1&coords=0.33145952866434314,-0.3983493644095738,4.809096622529252
        auto heartBeatFunc = [](float time){
            float k = M_PI * 4.0/3.0;
            return (float)
                (1
                + cos(fmod(time, k))                    // Long slow expansion
                * round(sin(fmod(time, k))/2+0.5)
                + cos(fmod(time, k)*3)                  // Short fast compression
                * (1-round(sin(fmod(time, k))/2+0.5))
                )/2;                                            // clamp to 0-1 range
        };

        // Grow in direction of normal and atrialPattern sample. Both move and scale splat up.
        // Techincally, normals should be transformed to world space by multiplying by world matrix, but there's no world matrix when we're working with a single model.
        float atrialGrowth = heartBeatFunc(time) * atrialPattern;
        float ventricularGrowth = heartBeatFunc(time-0.9f) * ventricularPattern; // offset ventricular cycle slightly
        
        glm::vec3 atrialPosModifier = *io.normal * atrialGrowth * 0.025f;
        glm::vec3 ventricularPosModifier = *io.normal * ventricularGrowth * 0.025f;

        glm::vec3 atrialScaleModifier = glm::vec3(atrialGrowth, atrialGrowth, atrialGrowth) * 0.0025f;
        glm::vec3 ventricularScaleModifier = glm::vec3(ventricularGrowth, ventricularGrowth, ventricularGrowth) * 0.0025f;
        
        *io.position = (*io.position) + atrialPosModifier + ventricularPosModifier;
        *io.scale = (*io.scale) + atrialScaleModifier + ventricularScaleModifier;

        // Darken areas that barely grow to simulate shadows. 
        //float totalGrowth = (atrialGrowth + ventricularGrowth)/2;
        // We floor from 0.5 to make sure only stuff that barely moves gets darkened
        // Techincally we could pull the colors into a HSV or HSL space to make better shadows, but implementing that will probably take too long.
        //io.sh[0] = io.sh[0] * (1 - max(0.0f, 0.5f-totalGrowth));
    }

    __device__ void CullHalf(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out)
    {
        // make one half of the model transparent.
        if(io.position->x < 0){
            *io.opacity = 0;

            // also make it tiny. This should make sure it doesn't contribute to any pixels during splat rendering, thereby reducing workload.
            *io.scale = glm::vec3(0);
        }
    }

    // The dissolve effect reimagined to better fit Gaussian splatting
    __device__ void GaussDissolve(ShShaderConstantInputs in, ShShaderModifiableInputs io, ShShaderOutputs out)
    {
        cudaTextureObject_t maskTex = in.d_textureManager->GetTexture("Cracks");

        // Grab the opacity from a mask texture
        float maskSample_xy = tex2D<float4>(maskTex, io.position->x, io.position->y).x;
        float maskSample_xz = tex2D<float4>(maskTex, io.position->x, io.position->z).x;
        float maskSample_yz = tex2D<float4>(maskTex, io.position->y, io.position->z).x;
        
        // combine masking from the 3 planes to create a 3d mask.
        float maskSample = maskSample_xy * maskSample_xz * maskSample_yz;

        // make the mask less gray
        maskSample = __saturatef((maskSample-0.125)*1.5);

        // repeat the effect over time
        float loadingSpeed = 0.25;
        float loopDuration = 3;
        float totalLoadProgression = fmod(in.time/1000 * loadingSpeed, loopDuration);
        
        // Start loading the model from the bottom. Offset loading with the mask
        float loadingPercent = __saturatef(totalLoadProgression - io.position->z + maskSample - 1);

        // As the splat loads, fade it in
        *io.opacity *= loadingPercent * loadingPercent * loadingPercent;

        // sart loading falling down into place from above during loading.
        // TODO: or from random direction! (use thread ID as seed for random direction.)
        glm::vec3 startPos = *io.position + glm::vec3(0, 0, glm::length(*io.scale) * 10);
        glm::vec3 currentPosition = glm::mix(startPos, *io.position, loadingPercent);
        *io.position = currentPosition;

        // Slightly tint the color towards bright blue as it loads.
        glm::vec3 targetfadeColor = glm::vec3(0.6,0.9,1);
        io.sh[0] = glm::mix(targetfadeColor, io.sh[0], loadingPercent);
    }


    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    __device__ ShShader defaultShader = &DefaultShShaderCUDA;
    __device__ ShShader expPosShader = &ExponentialPositionShaderCUDA;
    __device__ ShShader heartbeatShader = &HeartbeatShaderCUDA;
    __device__ ShShader cullHalf = &CullHalf;
    __device__ ShShader gaussDissolve = &GaussDissolve;

    std::map<std::string, int64_t> GetShShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // There doesn't seem to be any problem casting them back and forth though signed int64s.
        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(ShShader);
        
        // Copy device shader pointers to host map
        ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["ShDefault"] = (int64_t)h_defaultShader;

        ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        shaderMap["ExpPos"] = (int64_t)h_exponentialPositionShader;

        ShShader h_heartbeatShader;
        cudaMemcpyFromSymbol(&h_heartbeatShader, heartbeatShader, shaderMemorySize);
        shaderMap["Heartbeat"] = (int64_t)h_heartbeatShader;

        ShShader h_cullHalf;
        cudaMemcpyFromSymbol(&h_cullHalf, cullHalf, shaderMemorySize);
        shaderMap["CullHalf"] = (int64_t)h_cullHalf;

        ShShader h_gaussDissolve;
        cudaMemcpyFromSymbol(&h_gaussDissolve, gaussDissolve, shaderMemorySize);
        shaderMap["GaussDissolve"] = (int64_t)h_gaussDissolve;

        return shaderMap;
    }


    __global__ void ExecuteSHShaderCUDA(ShShader shader, int* d_splatIndexes, PackedShShaderParams packedParams){
        auto shaderInstance = cg::this_grid().thread_rank();
        if (shaderInstance >= packedParams.P)
            return;

        // Figure out which splat to execute on
        int idx = d_splatIndexes[shaderInstance];

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        ShShaderConstantInputs in(packedParams, idx);
        ShShaderModifiableInputs io(packedParams, idx);
        ShShaderOutputs out(packedParams, idx);

        // No need to dereference the shader function pointer.
        shader(in, io, out);
    }
}