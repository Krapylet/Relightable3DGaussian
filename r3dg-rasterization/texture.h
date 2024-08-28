#pragma once

#include <string>
#include <torch/extension.h>

namespace Texture{

    __global__ extern void PrintFirstPixel(cudaTextureObject_t texObj);

    // Creates a textureObject wrapper around the provided texture data and writes it to the texObjPtr
    //TODO: Test whether this pointer can be returned to python and used in another c call.
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData);

    void UnloadTexture(cudaTextureObject_t textureObject);

    // Encodes the name of a Pillow supported image mode to an int
    int encodeTextureMode(std::string mode);
    
    // Decodes the name of a Pillow supported image mode to a string
    std::string decodeTextureMode(int mode);

};