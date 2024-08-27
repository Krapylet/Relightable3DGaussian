#pragma once

#include <string>
#include <torch/extension.h>

namespace Texture{
    // Creates a textureObject wrapper around the provided texture data
    cudaTextureObject_t* CreateTexture(std::map<std::string, torch::Tensor> textureData);

    void UnloadTexture(cudaTextureObject_t textureObject);

    // Encodes the name of a Pillow supported image mode to an int
    int encodeTextureMode(std::string mode);
    
    // Decodes the name of a Pillow supported image mode to a string
    std::string decodeTextureMode(int mode);

};