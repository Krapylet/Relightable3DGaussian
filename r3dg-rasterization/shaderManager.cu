#include "shaderManager.h"
#include "utils/charOperations.h"

void ShaderManager::printAllAddresses(){
    printf("Printing %i addresses\n", h_shaderCount);
    for (size_t i = 0; i < h_shaderCount; i++)
    {
        printf("Address %i: %llu\n", i, h_shaderAddresses[i]);
    }
    
}

__global__ void printAddress(int64_t* addresses, int i){
    printf("Saving %llu shader address on GPU\n", addresses[i]);
}

ShaderManager::ShaderManager(std::map<std::string, int64_t> shaders){
    // initialize containers
    h_shaderCount = shaders.size();
    h_d_shaderAssociationMap = new int*[h_shaderCount];
    h_shaderInstanceCount = new int[h_shaderCount];
    h_shaderAddresses = new int64_t[h_shaderCount];
    cudaMalloc(&d_shaderInstanceCount, h_shaderCount * sizeof(int));
    cudaMalloc(&d_d_shaderNames, h_shaderCount * sizeof(char*));
    cudaMalloc(&d_shaderAddresses, h_shaderCount * sizeof(int64_t));
    cudaMalloc(&d_d_shaderAssociationMap, h_shaderCount * sizeof(int*));
    

    // Copy data
    int i = 0;
    for (auto [name, address]: shaders)
    {
        // copy names to device
        char* d_name;
        int nameLength = strlen(name.c_str()) + 1;
        cudaMalloc(&d_name, nameLength * sizeof(char));
        cudaMemcpy(d_name, name.c_str(), nameLength * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_d_shaderNames[i], &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // copy shader addresses
        h_shaderAddresses[i] = address;
        cudaMemcpy(&d_shaderAddresses[i], &address, sizeof(int64_t), cudaMemcpyHostToDevice);

        // initialize shader counts to 0
        int initValue = 0;
        cudaMemcpy(&d_shaderInstanceCount[i], &initValue, sizeof(int), cudaMemcpyHostToDevice);
        i++;
    }

    // copy shader count to device.
    cudaMalloc(&d_shaderCount, sizeof(int));
    cudaMemcpy(d_shaderCount, &h_shaderCount, sizeof(int), cudaMemcpyHostToDevice);

}

ShaderManager::~ShaderManager(){
    delete[] h_d_shaderAssociationMap;
    delete[] h_shaderInstanceCount;
    delete[] h_shaderAddresses;
    cudaFree(d_shaderCount);
    cudaFree(d_shaderInstanceCount);
    cudaFree(d_d_shaderNames);
    cudaFree(d_shaderAddresses);
    cudaFree(d_d_shaderAssociationMap);
}

__device__ int ShaderManager::GetIndexOfShader(char* targetShaderName){
    for (size_t i = 0; i < *d_shaderCount; i++)
    {
        // Check if the name in the lookup table is the same as the target name
        char* currentShaderName = d_d_shaderNames[i];
        bool shaderHasBeenFound = charsAreEqual(currentShaderName, targetShaderName);
    
        if(shaderHasBeenFound){
            return i;
        }
    }

    // If no texture was found with the given name, return the default error texture
    printf("Warning: Could not find shader '%s'\n", targetShaderName);
    return 0;
}

__device__ int64_t ShaderManager::GetShader(int shaderIndex){
    return d_shaderAddresses[shaderIndex];
}

__device__ int64_t ShaderManager::GetShader(char* shaderName){
    return d_shaderAddresses[GetIndexOfShader(shaderName)];
}