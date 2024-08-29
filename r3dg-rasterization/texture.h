#pragma once

#include <string>
#include <torch/extension.h>

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

    __global__ extern void PrintFirstPixel(cudaTextureObject_t texObj);

    // Creates a textureObject wrapper around the provided texture data and writes it to the texObjPtr
    //TODO: Test whether this pointer can be returned to python and used in another c call.
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData);

    void UnloadTexture(cudaTextureObject_t* textureObject);

    // Encodes the name of a Pillow supported image mode to an int
    int EncodeTextureMode(std::string mode);

    int EncodeWrapMode(std::string mode);
};