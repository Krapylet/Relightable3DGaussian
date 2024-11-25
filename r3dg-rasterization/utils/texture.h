#pragma once

#include <string>
#include "includeTorch.cu"

// A wrapper class around the indirect lookup table. Mostly just used to retrieve textures.
namespace Texture
{
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

    // A Class in a namespace
    class TextureManager
    {
    private:
        // Device stored variables:
        int *d_texCount;
        char** d_textureNames;
        cudaTextureObject_t* d_textureObjects;
        cudaTextureObject_t* d_errorTexture;
        // additional information about the textures could be added here as new arrays. Or a textureWrapper class could be created, that stored name and texture and other information.
        
        // Host stored variables:  asd
        // (None so far)
    public:
        __host__ TextureManager();

        // Allocates and Uploads an array an array of textures onto the GPU so that textures can be looked up by in the shaders.
        __host__ void SetTextures(std::vector<std::string> names, std::vector<int64_t> textureObjects);
                
        // Deallocates all textures on the device (except the error texture)
        __host__ void UnloadTextures();

        // Allocates and uploads an errot texture to the device.
        __host__ void SetErrorTexture(cudaTextureObject_t* errorTexture);

        //Deallocates the error texture on the device.
        __host__ void UnloadErrorTexture();

        // Returns the error texture.
        __device__ cudaTextureObject_t GetErrorTexture();
                
        // Loops through each loaded texture name and checks if it matches the given name before returning the associated texture. Is pretty slow, so cache the result.
        // Returns the error texture if no texture is found.
        __device__ cudaTextureObject_t GetTexture(char* targetTextureName);

    };

    // Creates a textureObject wrapper around the provided texture data and allocates persistent memory to store it in.
    int64_t AllocateTexture(std::map<std::string, torch::Tensor> textureData);

    // Loads the texture name and texture object vectors onto the GPU.
    int64_t UploadTexturesToDevice(std::vector<std::string> names, std::vector<int64_t> textureObjects, int64_t errorTex);

    // Creates a textureObject wrapper around the provided texture data and writes it to the texObjPtr
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData);

    void UnloadTexture(cudaTextureObject_t* textureObject);

    // Encodes the name of a Pillow supported image mode to an int
    int EncodeTextureMode(std::string mode);

    int EncodeWrapMode(std::string mode);
};