#pragma once

#include <string>
#include <torch/extension.h>
//#include <thrust/device_vector.h>
//#include <thrust/host_vector.h>
//#include <string>

namespace Texture{
    enum TextureMode {
        Unknown = -1,
        One = 0,
        L = 1,
        P = 2,
        RGB = 3,
        RGBA = 4,
        CMYK = 5,
        YCbCr = 6,
        LAB = 7,
        HSV = 8,
        I = 9,
        F = 10
    };

    // Creates a textureObject wrapper around the provided texture data and allocates persistent memory to store it in.
    int64_t AllocateTexture(std::map<std::string, torch::Tensor> textureData);

    // Loads the texture name and texture object vectors onto the GPU.
    std::pair<int64_t, int64_t> LoadDeviceTextureVectors(std::vector<std::string> names, std::vector<cudaTextureObject_t*> textureObjects);

    // Creates a textureObject wrapper around the provided texture data and writes it to the texObjPtr
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData);

    void UnloadTexture(cudaTextureObject_t* textureObject);

    // Encodes the name of a Pillow supported image mode to an int
    int EncodeTextureMode(std::string mode);

    int EncodeWrapMode(std::string mode);


    int64_t InitializeTextureMaps(const std::map<std::string, std::map<std::string, std::map<std::string, torch::Tensor>>>& shaderTextureTensorMaps);
    void UnloadTextureMaps (int64_t shaderTextureMaps_mapPtr);

    /// Debug methods. Don't use
    void PrintFromFirstTexture (int64_t shaderTextureMaps_mapPtr);
    __global__ extern void PrintFirstPixel(cudaTextureObject_t texObj);
    void PrintFromWrappedTexture(int64_t texObj_int64_t_ptr);


    int64_t AllocateVariable();
    void PrintVariable (int64_t allocedPointer_intPtr);
    void DeleteVariable(int64_t allocedPointer_intPtr);
};