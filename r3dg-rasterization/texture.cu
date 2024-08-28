#include "texture.h"
#include "third_party/lodepng/lodepng.h"
#include "cuda_rasterizer/auxiliary.h"
#include <iostream>
#include <stdexcept>
#include <map>

namespace Texture
{
    __global__ void PrintFirstPixel(cudaTextureObject_t texObj){
        //TODO: Wrap a class around the texture object that automatically gets the correct type of value from the texture based on the texture mode.
        float4 cudaTexel = tex2D<float4>(texObj, 0, 0);
        printf("Cuda reading RGBA value of first texel: %f,%f,%f,%f\n", cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
    }

    // Allocates a new array from the input array where every 4th index is a padded value of 1. The input pointer is overwritten with the pointer to the new array.
    // The array is 4/3rds the length of the input array. Remember to delete the allocated array.
    __global__ void CreatPaddedArrayFromBase(float* src, float* dest, int width, int height){
        // Increase count by 1/3rd to make room for the 4th channel
        for (size_t padded_i = 0; padded_i < width*height*4; padded_i++)
        {
            int i = 0;
            bool padCurrentIndex = (padded_i + 1) % 4 == 0;
            if (padCurrentIndex)
            {
                dest[padded_i] = 1;
            }
            else{
                dest[padded_i] = src[i];
                i++;
            }
        }
    }

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
    int encodeTextureMode(std::string mode){
        if(mode == "1")
            return 0;
        else if ( mode == "L")
            return 1;
        else if ( mode == "P")
            return 2;
        else if ( mode == "RGB")
            return 3;
        else if ( mode == "RGBA")
            return 4;
        else if ( mode == "CMYK")
            return 5;
        else if ( mode == "YCbCr")
            return 6;
        else if ( mode == "LAB")
            return 7;
        else if ( mode == "HSV")
            return 8;
        else if ( mode == "I")
            return 9;
        else if ( mode == "F")
            return 10;

        return -1;
    }

    std::string decodeTextureMode(int mode){
        if(mode == 0)
            return "1";
        else if ( mode == 1)
            return "L";
        else if ( mode == 2)
            return "P";
        else if ( mode == 3)
            return "RGB";
        else if ( mode == 4)
            return "RGBA";
        else if ( mode == 5)
            return "CMYK";
        else if ( mode == 6)
            return "YCbCr";
        else if ( mode == 7)
            return "LAB";
        else if ( mode == 8)
            return "HSV";
        else if ( mode == 9)
            return "I";
        else if ( mode == 10)
            return "F";

        return "Error: Unkown texture mode encoding '" + std::to_string(mode) + "'.";
    }

    // Creates a textureObject wrapper around the provided texture data
    // Adapted from the lodepng decoding example example at https://github.com/lvandeve/lodepng/blob/master/examples/example_decode.cpp
    // and the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
    void CreateTexture(cudaTextureObject_t* texObjPtr, std::map<std::string, torch::Tensor> textureData){

        // extract all the texture data
        int height = textureData["height"].const_data_ptr<int>()[0];
        int width = textureData["width"].const_data_ptr<int>()[0];
        std::string mode = decodeTextureMode(textureData["mode"].const_data_ptr<int>()[0]);
        float* pixelData = textureData["pixelData"].contiguous().cuda().mutable_data_ptr<float>();
        int pixelDataCount = textureData["pixelData"].numel();

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
        if(mode == "1" || mode == "L" || mode == "P" || mode == "I" || mode == "F"){
            channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float1);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
        }
        else if ( mode == "RGBA" || mode == "CMYK")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
        }
        else if ( mode == "RGB" || mode == "YCbCr")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3, so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            CreatPaddedArrayFromBase<<<1,1>>>(pixelData, paddedData, width, height);         //TODO: accelerate this kernel with more threads 
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        else if (mode == "LAB" || mode == "HSV")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
            channelByteWidth = width * sizeof(float4);
            paddedDataSize = height * channelByteWidth;
            texDesc.filterMode = cudaFilterModePoint;  // Linear filtering is only available for floats 
            texDesc.readMode = cudaReadModeNormalizedFloat; //Notice: This might actually not work with 32 bit channels. Only 16 and 8 bit channels. But I haven't tested it.

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3m so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataSize);
            CreatPaddedArrayFromBase<<<1,1>>>(pixelData, paddedData, width, height);         //TODO: accelerate this kernel with more threads 
            cudaDeviceSynchronize();
            pixelData = paddedData; // Overwrite the original data pointer. Remember to free the memory by the end of the function.
        }
        // Specify remaining texture object parameters
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;         //TODO: allow the wrap mode to be set based on a texture import setting
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
        
        if ( mode == "RGB" || mode == "YCbCr" || mode == "LAB" || mode == "HSV")
        {   
            // If we had to copy and pad the data of a 3-value format with a 4th value before the data was copied to a cudaArray,
            // we have to free the memory used to create the temporary padded version. 
            cudaFree(pixelData);
        }
 
        // TODO: Make sure to keep track of which memory we need clean up at the end of this function, and at the end of this frame.
    }

    // TODO:unload texture
    // Free device memory at the end of the frame
    // cudaResourceDesc* resDesc_DeleteThisPart;
    // cudaGetTextureObjectResourceDesc(resDesc_DeleteThisPart, texObj);
    //cudaFreeArray(resDesc_DeleteThisPart->res.array.array);
}



