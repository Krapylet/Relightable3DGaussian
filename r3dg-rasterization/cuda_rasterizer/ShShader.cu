#include "shShader.h"
#include "config.h"
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
    __device__ ShShaderParams::ShShaderParams(PackedShShaderParams p, int idx):
        time(p.time), dt(p.dt),
        scale_modifier(p.scale_modifier),
		grid(p.grid),
		viewmatrix(p.viewmatrix),
		viewmatrix_inv(p.viewmatrix_inv),
		projmatrix(p.projmatrix),
		projmatrix_inv(p.projmatrix_inv),
        camera_position({p.viewmatrix_inv[12], p.viewmatrix_inv[13], p.viewmatrix_inv[14]}),
		W(p.W), H(p.H),
		focal_x(p.focal_x), focal_y(p.focal_y),
		tan_fovx(p.tan_fovx), tan_fovy(tan_fovy),
        deg(p.deg), max_coeffs(p.max_coeffs),
        
        // Precomputed pr. splat 'texture' information from the neilf pbr decomposition
        roughness (p.features[idx * p.S + 0]),
		metallic (p.features[idx * p.S + 1]),
		incident_visibility (p.features[idx * p.S + 2]),
		color_brdf ({p.features[idx * p.S + 3], p.features[idx * p.S + 4], p.features[idx * p.S + 5]}),
		normal ({p.features[idx * p.S + 6], p.features[idx * p.S + 7], p.features[idx * p.S + 8]}),
		color_base ({p.features[idx * p.S + 9], p.features[idx * p.S + 10], p.features[idx * p.S + 11]}),
		incident_light ({p.features[idx * p.S + 12], p.features[idx * p.S + 13], p.features[idx * p.S + 14]}),
		local_incident_light ({p.features[idx * p.S + 15], p.features[idx * p.S + 16], p.features[idx * p.S + 17]}),
		global_incident_light ({p.features[idx * p.S + 18], p.features[idx * p.S + 19], p.features[idx * p.S + 20]}),

        d_textureManager(p.d_textureManager),
        
		//input/output   -   contains values when the method is called that can be changed.
		position(p.positions + idx),
		scale(p.scales + idx),
		rotation(p.rotations + idx),
		opacity(p.opacities + idx),
		sh(p.shs + idx * p.max_coeffs), // could also be calculated as idx + (p.deg + 1)^2
        
        // output
        stencil_val(p.stencil_vals + idx)
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ static void DefaultShShaderCUDA(ShShaderParams p)
    {
        // Default shader doesn't need to set any values.
    }

    __device__ static void ExponentialPositionShaderCUDA(ShShaderParams p)
    {
        // multiply sh position and scale by y coordinate
        float posX = abs((*p.position).x);
        float posY = abs((*p.position).y);
        float posZ = abs((*p.position).z);
        float dist = glm::length(*p.position);
        
        *p.scale = glm::vec3((*p.scale).x * posY, (*p.scale).y * 2, (*p.scale).z) * posY;
        *p.position = glm::vec3((*p.position).x * posY, (*p.position).y * 2, (*p.position).z) * posY;
    }

    // A shader that makes an object pulse with a heartbeat
    // The heart has two concurrent beats that we model here.
    // Written as SH shader because this is where we can update positions and scales.
    __device__ static void HeartbeatShaderCUDA(ShShaderParams p)
    {
        // Sample the texture contiaing the pattern for growth with the atreal pulse.
        cudaTextureObject_t atrialTex = p.d_textureManager->GetTexture("Turbulence");
        float atrialSampleXY = tex2D<float4>(atrialTex, p.position->x, p.position->y).x;
        float atrialSampleXZ = tex2D<float4>(atrialTex, p.position->x, p.position->z).x;
        float atrialSampleYZ = tex2D<float4>(atrialTex, p.position->y, p.position->z).x;
        float atrialPattern = (atrialSampleXY + atrialSampleXZ + atrialSampleYZ) / 3;

        // Sample for ventricular pattern
        cudaTextureObject_t ventricularTex = p.d_textureManager->GetTexture("Craters");
        float ventricularSampleXY = 1-tex2D<float4>(ventricularTex, p.position->x, p.position->y).x;
        float ventricularSampleXZ = 1-tex2D<float4>(ventricularTex, p.position->x, p.position->z).x;
        float ventricularSampleYZ = 1-tex2D<float4>(ventricularTex, p.position->y, p.position->z).x;
        float ventricularPattern = (ventricularSampleXY + ventricularSampleXZ + ventricularSampleYZ) / 3;
 
        // Create a wave that pulses out from the center of the object with values from 0-1.
        // Technically, this should instead expand from the center of the object, but since the objects don't currently have a pivot when
        // when only rendinering a single object, I'm using the origin of the world instead.
        float pulsePeriod = 1;
        float distInfluence = -0.5;
        float splatDist = glm::length(*p.position);
        float time = p.time/1000/pulsePeriod + splatDist * distInfluence;

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
        
        glm::vec3 atrialPosModifier = p.normal * atrialGrowth * 0.025f;
        glm::vec3 ventricularPosModifier = p.normal * ventricularGrowth * 0.025f;

        glm::vec3 atrialScaleModifier = glm::vec3(atrialGrowth, atrialGrowth, atrialGrowth) * 0.0025f;
        glm::vec3 ventricularScaleModifier = glm::vec3(ventricularGrowth, ventricularGrowth, ventricularGrowth) * 0.0025f;
        
        *p.position = (*p.position) + atrialPosModifier + ventricularPosModifier;
        *p.scale = (*p.scale) + atrialScaleModifier + ventricularScaleModifier;

        // Darken areas that barely grow to simulate shadows. 
        //float totalGrowth = (atrialGrowth + ventricularGrowth)/2;
        // We floor from 0.5 to make sure only stuff that barely moves gets darkened
        // Techincally we could pull the colors into a HSV or HSL space to make better shadows, but implementing that will probably take too long.
        //p.sh[0] = p.sh[0] * (1 - max(0.0f, 0.5f-totalGrowth));
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    //TODO: Instead of storing shaders in individual variables, store them in a __device__ const map<ShShaderName, ShShader> 
    __device__ const ShShader defaultShader = &DefaultShShaderCUDA;
    __device__ const ShShader expPosShader = &ExponentialPositionShaderCUDA;
    __device__ const ShShader heartbeatShader = &HeartbeatShaderCUDA;

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

        return shaderMap;
    }

    // ONETIME USE FUNCTION USED TO DEBUG. ALLOCATES THE RETURN ARRAY. REMEMBER TO FREE AFTER USE.
    // Returns an array in device memory containing addresses to device shader functions.
    int64_t* GetShShaderAddressArray(){
        // Array is assembled on CPU before being sent to device. Addresses themselves are in device space.
        int shaderCount = 3;
        int64_t* h_shaderArray = new int64_t[shaderCount];
        size_t shaderMemorySize = sizeof(ShShader);

        ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        h_shaderArray[0] = (int64_t)h_defaultShader;

        ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        h_shaderArray[1] = (int64_t)h_exponentialPositionShader;

        ShShader h_heartbeatShader;
        cudaMemcpyFromSymbol(&h_heartbeatShader, heartbeatShader, shaderMemorySize);
        h_shaderArray[2] = (int64_t)h_heartbeatShader;

        // copy the array to device
        int64_t* d_shaderArray;
        cudaMalloc(&d_shaderArray, sizeof(int64_t)*shaderCount);
        cudaMemcpy(d_shaderArray, h_shaderArray, shaderMemorySize * shaderCount, cudaMemcpyDefault);
        

        // Delete temporary host array.
        delete[] h_shaderArray;
        return d_shaderArray;
    }

    __global__ void ExecuteShader(ShShader* shaders, PackedShShaderParams packedParams){
        // calculate index for the spalt.
        auto idx = cg::this_grid().thread_rank();
        if (idx >= packedParams.P)
            return;

        // Unpack shader parameters into a format that is easier to work with. Increases memory footprint as tradeoff.
        // Could easily be optimized away by only indexing into the params inside the shader, but for now I'm prioritizing ease of use.
        ShShaderParams params(packedParams, idx);

        // No need to dereference the shader function pointer.
        shaders[idx](params);
    }



    /// --------------------------- Debug methods ------------------------

    /*
    __global__ void TestFunctionPointerMapCUDA(){
        printf("CUDA - Declaring function pointer map");
        std::map<ShShader, int> functionPointerMap;
        printf("CUDA - Assigning values");
        functionPointerMap[defaultShader] = 1;
        functionPointerMap[expPosShader] = 2;
        printf("CUDA - Retriving values");
        int result = functionPointerMap[defaultShader] + functionPointerMap[expPosShader];
        printf("CUDA - Retreved values sucessfully");
    }
    
    void TestFunctionPointerMap(){
        printf("Declaring function pointer map");
        std::map<ShShader, int> functionPointerMap;
        printf("Assigning values");
        functionPointerMap[defaultShader] = 1;
        functionPointerMap[expPosShader] = 2;
        printf("Retriving values");
        int result = functionPointerMap[defaultShader] + functionPointerMap[expPosShader];
        printf("Retreved values sucessfully");

        printf("Repeating experiment on device");
        TestFunctionPointerMapCUDA<<<1,1>>>();
        cudaDeviceSynchronize();
        printf("Experiment on device done");
    }
    */
}