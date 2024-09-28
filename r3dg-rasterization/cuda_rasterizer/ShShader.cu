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

    __device__ static void DiffuseShader(ShShaderParams p){
        cudaTextureObject_t grainyTexture = p.d_textureManager->GetTexture("Grainy");
        
        float opacity = tex2D<float4>(grainyTexture, p.position->x, p.position->y).w;

        // Make sure we don't get negative opacity
        opacity = __saturatef(opacity);

        *p.opacity = opacity;
    }

    ///// Assign all the shaders to their short handles.
    // we need to keep them in constant device memory for them to stay valid when passed to host.
    //TODO: Instead of storing shaders in individual variables, store them in a __device__ const map<ShShaderName, ShShader> 
    __device__ const ShShader defaultShader = &DefaultShShaderCUDA;
    __device__ const ShShader expPosShader = &ExponentialPositionShaderCUDA;
    __device__ const ShShader diffuseShader = &DiffuseShader;

    IndirectMap<char*, ShShader>* GetShShaderAddressMap(){
        std::vector<char*> shaderNames;
        std::vector<ShShader> shaderFunctionPointers;
        size_t shaderMemorySize = sizeof(ShShader);
        
        // Copy device shader pointers to host map
        ShShader::ShShader h_defaultShader;
        cudaMemcpyFromSymbol(&h_defaultShader, defaultShader, shaderMemorySize);
        shaderNames.push_back("ShDefault");
        shaderFunctionPointers.push_back(h_defaultShader);

        ShShader::ShShader h_exponentialPositionShader;
        cudaMemcpyFromSymbol(&h_exponentialPositionShader, expPosShader, shaderMemorySize);
        shaderNames.push_back("ExpPos");
        shaderFunctionPointers.push_back(h_exponentialPositionShader);

        ShShader::ShShader h_diffuseShader;
        cudaMemcpyFromSymbol(&h_diffuseShader, diffuseShader, shaderMemorySize);
        shaderNames.push_back("Diffuse");
        shaderFunctionPointers.push_back(h_diffuseShader);

        IndirectMap<char*, ShShader>* shShaderMap = new IndirectMap<char*, ShShader>(shaderNames, shaderFunctionPointers);

        return shShaderMap;
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