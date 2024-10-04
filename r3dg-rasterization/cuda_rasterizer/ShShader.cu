#include "shShader.h"
#include "config.h"
#include <iostream>
#include <cooperative_groups.h>
#include "../utils/texture.h"
#ifndef GLM_FORCE_CUDA
#define GLM_FORCE_CUDA
#endif
#include <glm/glm.hpp>


namespace cg = cooperative_groups;

namespace ShShader
{
    //TODO: we can't actually have strings on device, so we have to create enums as aliases for the shader names.
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
        d_textureManager(p.d_textureManager),
        
		//input/output   -   contains values when the method is called that can be changed.
		position(p.positions + idx),
		scale(p.scales + idx),
		rotation(p.rotations + idx),
		opacity(p.opacities + idx),
		sh(p.shs + idx * p.max_coeffs) // could also be calculated as idx + (p.deg + 1)^2
        {
		// for now we're not actually doing anyting in the constuctior other than initializing the constants.
    }

    __device__ static void DefaultShShaderCUDA(ShShaderParams p)
    {
        // Set output color
        //*p.out_color = (*p.color_SH);
    }

    __device__ static void ExponentialPositionShaderCUDA(ShShaderParams p)
    {
        // multiply sh position and scale by y coordinate
        float posX = abs((*p.position).x);
        float posY = abs((*p.position).y);
        float posZ = abs((*p.position).z);
        float dist = (*p.position).length();
        
        *p.scale = glm::vec3((*p.scale).x * posY, (*p.scale).y * 2, (*p.scale).z) * posY;
        *p.position = glm::vec3((*p.position).x * posY, (*p.position).y * 2, (*p.position).z) * posY;
    }

    __device__ static void DissolveShader(ShShaderParams p){
        /*
        cudaTextureObject_t grainyTexture = p.d_textureManager->GetTexture("Black");

        float texSample = tex2D<float4>(grainyTexture, 3.0/4.0+0.005, 3.0/4.0+0.005).x;

        // Make sure we don't get negative opacity
        // goes back and forth between 0-1
        float opacityPercent = (cosf(p.time/500) + 1)/2;

        float originalOpacity = *p.opacity;

        float opacity = __saturatef((1 + texSample) * opacityPercent * originalOpacity);

        *p.opacity = texSample* originalOpacity;

        // gå fra 1 til 0 over tid med opacity. 
        // ved 1, gang 
        */
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    //TODO: Instead of storing shaders in individual variables, store them in a __device__ const map<ShShaderName, ShShader> 
    __device__ const ShShader defaultShader = &DefaultShShaderCUDA;
    __device__ const ShShader expPosShader = &ExponentialPositionShaderCUDA;
    __device__ const ShShader disolveShader = &DissolveShader;

    std::map<std::string, int64_t> GetShShaderAddressMap(){
        // we cast pointers to numbers since most pointers aren't supported by pybind
        // Device function pointers seem to be 8 bytes long (at least on the devlopment machine with a GTX 2080 and when compiling to 64bit mode)
        // There doesn't seem to be any problem casting them back and forth though signed int64s.
        std::map<std::string, int64_t> shaderMap;
        size_t shaderMemorySize = sizeof(ShShader);
        
        // Copy device shader pointers to host map
        ShShader::ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderMap["ShDefault"] = (int64_t)h_defaultShader;

        ShShader::ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        shaderMap["ExpPos"] = (int64_t)h_exponentialPositionShader;

        ShShader::ShShader h_disolveShader;
        cudaMemcpyFromSymbol(&h_disolveShader, disolveShader, shaderMemorySize);
        shaderMap["Diffuse"] = (int64_t)h_disolveShader;

        return shaderMap;
    }

    // ONETIME USE FUNCTION USED TO DEBUG. ALLOCATES THE RETURN ARRAY. REMEMBER TO FREE AFTER USE.
    // Returns an array in device memory containing addresses to device shader functions.
    int64_t* GetShShaderAddressArray(){
        // Array is assembled on CPU before being sent to device. Addresses themselves are in device space.
        int shaderCount = 3;
        int64_t* h_shaderArray = new int64_t[shaderCount];
        size_t shaderMemorySize = sizeof(ShShader);

        ShShader::ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        h_shaderArray[0] = (int64_t)h_defaultShader;

        ShShader::ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        h_shaderArray[1] = (int64_t)h_exponentialPositionShader;

        ShShader::ShShader h_disolveShader;
        cudaMemcpyFromSymbol(&h_disolveShader, disolveShader, shaderMemorySize);
        h_shaderArray[2] = (int64_t)h_disolveShader;

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

        if (idx == 1)
            printf("Time: %f, cosTime: %f\n", params.time, (cosf(params.time/1000) + 1)/2);

        // No need to dereference the shader function pointer.
        shaders[idx](params);

        //DefaultShShaderCUDA(params);
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