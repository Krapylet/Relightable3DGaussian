#pragma once

#include <string>
#include <map>


// a class for mapping 3D gaussians to a type of shader.
class ShaderManager
{
public:
    int h_shaderCount;
    int* h_shaderInstanceCount;
    int** h_d_shaderAssociationMap;
    int64_t* h_shaderAddresses;

    
    // Data is mirrored on device
    int* d_shaderCount; // not an array.
    int* d_shaderInstanceCount;
    int** d_d_shaderAssociationMap;
    int64_t* d_shaderAddresses;

    char** d_d_shaderNames;


    ShaderManager(std::map<std::string, int64_t> shaders);
    ~ShaderManager();

    __device__ int GetIndexOfShader(char* shaderName);
    __device__ int64_t GetShader(char* shaderName);
    __device__ int64_t GetShader(int shaderIndex);
};