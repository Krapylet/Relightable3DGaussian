#include "texture.h"
#include "third_party/lodepng/lodepng.h"
#include "cuda_rasterizer/auxiliary.h"
#include <iostream>
#include <stdexcept>
#include <map>
#include <cooperative_groups.h>

namespace Texture
{
    __global__ void PrintFirstPixel(cudaTextureObject_t texObj){
        float4 cudaTexel = tex2D<float4>(texObj, 0, 0);
        printf("Cuda reading RGBA value of first texel: %f,%f,%f,%f\n", cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
    }

    // Allocates a new array from the input array where every 4th index is a padded value of 1. The input pointer is overwritten with the pointer to the new array.
    // The array is 4/3rds the length of the input array. Remember to delete the allocated array.
    __global__ void CreatPaddedArrayFromBase(float* src, float* dest, int paddedDataCount){
        auto padded_i = cooperative_groups::this_grid().thread_rank();
        if (padded_i >= paddedDataCount)
            return;

        bool padCurrentIndex = (padded_i + 1) % 4 == 0;
        if (padCurrentIndex)
        {
            dest[padded_i] = 1;
        }
        else{
            // paddedDataCount = height * width * 4
            // srcDataCount = height * width * 3
            int src_i = ceil(padded_i*0.75);
            dest[padded_i] = src[src_i];
        }
    }

    int EncodeTextureMode(std::string mode){
        if(mode == "1")
            return TextureMode::One;
        else if ( mode == "L")
            return TextureMode::L;
        else if ( mode == "P")
            return TextureMode::P;
        else if ( mode == "RGB")
            return TextureMode::RGB;
        else if ( mode == "RGBA")
            return TextureMode::RGBA;
        else if ( mode == "CMYK")
            return TextureMode::CMYK;
        else if ( mode == "YCbCr")
            return TextureMode::YCbCr;
        else if ( mode == "LAB")
            return TextureMode::LAB;
        else if ( mode == "HSV")
            return TextureMode::HSV;
        else if ( mode == "I")
            return TextureMode::I;
        else if ( mode == "F")
            return TextureMode::F;

        return TextureMode::Unknown;
    }

    // Encodes string to cuda enum. Possible modes are:
    // - "Wrap": UVs outside the range wraps back around from the other side, repeating the texture.
    // - "Mirror": UBs outside the range wraps back from the same side, repeating the texture, but mirrored.
    // - "Clamp": UVs outside the range are clamped back into the range of the texture
    // - "Border": UVs outside the range return 0.
    int EncodeWrapMode(std::string mode){
        if(mode == "Border")
            return cudaAddressModeBorder;
        if(mode == "Clamp")
            return cudaAddressModeClamp;
        if(mode == "Mirror")
            return cudaAddressModeMirror;
        if(mode == "Wrap")
            return cudaAddressModeWrap;

        return -1;
    }

    // Creates a textureObject wrapper around the provided texture data
    // Adapted from the lodepng decoding example example at https://github.com/lvandeve/lodepng/blob/master/examples/example_decode.cpp
    // and the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData){
        // extract all the texture data
        int height = textureData["height"].const_data_ptr<int>()[0];
        int width = textureData["width"].const_data_ptr<int>()[0];
        TextureMode mode = static_cast<TextureMode>(textureData["encoding_mode"].const_data_ptr<int>()[0]);
        float* pixelData = textureData["pixelData"].contiguous().cuda().mutable_data_ptr<float>();
        int pixelDataCount = textureData["pixelData"].numel();
        const cudaTextureAddressMode* addressModes = (const cudaTextureAddressMode*)textureData["wrap_modes"].const_data_ptr<int>();

        /*
        Create a channel descriptor appropriate to the image mode.
        Pillow supported channel modes:
        1 (1-bit pixels, black and white, stored with one pixel per byte)
        L (8-bit pixels, grayscale)
        P (8-bit pixels, mapped to any other mode using a color palette)
        RGB (3x8-bit pixels, true color)
        RGBA (4x8-bit pixels, true color with transparency mask)
        CMYK (4x8-bit pixels, color separation)
        YCbCr (3x8-bit pixels, color video format)
        Note that this refers to the JPEG, and not the ITU-R BT.2020, standard
        LAB (3x8-bit pixels, the L*a*b color space)
        HSV (3x8-bit pixels, Hue, Saturation, Value color space)
        Hue’s range of 0-255 is a scaled version of 0 degrees <= Hue < 360 degrees
        I (32-bit signed integer pixels)
        F (32-bit floating point pixels)
        https://pillow.readthedocs.io/en/stable/handbook/concepts.html
        */
        cudaChannelFormatDesc channelDesc;
        int paddedDataSize;
        int channelByteWidth;
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        if(mode == TextureMode::One || mode == TextureMode::L || mode == TextureMode::P || mode == TextureMode::I || mode == TextureMode::F){
            channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float1);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
        }
        else if ( mode == TextureMode::RGBA || mode == TextureMode::CMYK)
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
        }
        else if ( mode == TextureMode::RGB || mode == TextureMode::YCbCr)
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3, so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            int paddedDataCount = width*height*4;
            CreatPaddedArrayFromBase<<<(paddedDataCount + 255) / 256, 256>>>(pixelData, paddedData, paddedDataCount);
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        else if (mode == TextureMode::LAB|| mode == TextureMode::HSV)
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModePoint;  // Linear filtering is only available for floats 
            texDesc.readMode = cudaReadModeNormalizedFloat; //Notice: This might actually not work with 32 bit channels. Only 16 and 8 bit channels. But I haven't tested it.

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3m so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            int paddedDataCount = width*height*4;
            CreatPaddedArrayFromBase<<<(paddedDataCount + 255) / 256, 256>>>(pixelData, paddedData, paddedDataCount);
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        // Specify remaining texture object parameters
        texDesc.addressMode[0] = addressModes[0];
        texDesc.addressMode[1] = addressModes[1];
        texDesc.normalizedCoords = 1;                           //TODO: allow the coordinate mode ot be set based on a texture import setting

        // Allocate CUDA array in device memory
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        // Set pitch of the source (the width in memory in bytes of the 2D array pointed to by src, including padding). We dont have any padding, so it's just equal to the byte width.
        const size_t pitch = channelByteWidth;
        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, pixelData, pitch, channelByteWidth, height, cudaMemcpyDeviceToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
        
        // Create texture object, which is used as a wrapper to access the cuda Array with the actual image data.
        checkCudaErrors(cudaCreateTextureObject(texObjPtr, &resDesc, &texDesc, NULL));
        
        if ( mode == TextureMode::RGB || mode == TextureMode::YCbCr || mode == TextureMode::LAB || mode == TextureMode::HSV)
        {   
            // If we had to copy and pad the data of a 3-value format with a 4th value before the data was copied to a cudaArray,
            // we have to free the memory used to create the temporary padded version. 
            cudaFree(pixelData);
        }
 
        // TODO: Make sure to keep track of which memory we need clean up at the end of this function, and at the end of this frame.
    }

    // Frees the underlying cudaArray that the textureObject is wrapped around
    void UnloadTexture(cudaTextureObject_t textureObject){
        cudaResourceDesc* resDesc;
        std::cout << "Getting resource desc" << std::endl;
        checkCudaErrors(cudaGetTextureObjectResourceDesc(resDesc, textureObject));
        std::cout << "Freeing cudaArray" << std::endl;
        checkCudaErrors(cudaFreeArray(resDesc->res.array.array));
        std::cout << "Memory Freed!" << std::endl;
    }
}



