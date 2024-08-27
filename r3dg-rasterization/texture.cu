#include "texture.h"
#include "third_party/lodepng/lodepng.h"
#include "cuda_rasterizer/auxiliary.h"
#include <iostream>
#include <stdexcept>
#include <map>

namespace Texture
{
    __global__ void PrintFirstPixel(cudaTextureObject_t texObj){
        uchar4 cudaTexel = tex2D<uchar4>(texObj, 0, 0);
        printf("Cuda reading RGBA value of first texel: %d,%d,%d,%d\n", cudaTexel.x, cudaTexel.y, cudaTexel.z, cudaTexel.w);
    }


    // load texture
    // returns poiner to cuda textureObject? or just raw pointer to image data?
    // Adapted from the lodepng decoding example example at https://github.com/lvandeve/lodepng/blob/master/examples/example_decode.cpp
    // and the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
    // remember to free the 
    cudaTextureObject_t* LoadTexture(std::string texturePath){
        
        //TODO: Write support for both RGB and RGBA pngs.

        std::vector<unsigned char> imageVec;
        unsigned int width, height;

        // Image is loaded as an array of interleaved chars on the format R1, G1, B1, A1, R2...
        // Each char has a range of 256 (8 bits) and represents either R,G,B or A, totalling 32 bits pr. pixel.
        // It seems that decoded RGB pictures automatically get an opaque A channel interleaved.
        unsigned int error = lodepng::decode(imageVec, width, height, texturePath);

        // Print any error that may appear during loading.
        if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        std::cout << "width: " << width << ", height: " << height << "\n" << std::endl;
        std::cout << "Values read in image: " << imageVec.size() << std::endl;
        std::cout << "lodepng reading RGBA value of first pixel: " << (int)imageVec[0] << "," << (int)imageVec[1] << "," << (int)imageVec[2] << "," << (int)imageVec[3] << "," << std::endl;
    
        // cast image vector to a char4 array so we can store it on cuda device as a texture
        uchar4* image_data = (uchar4*)&imageVec[0];

        printf("Host reading RGBA value of first texel as char4: %d,%d,%d,%d\n", image_data->x, image_data->y, image_data->z, image_data->w);

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        // Set pitch of the source (the width in memory in bytes of the 2D array pointed
        // to by src, including padding), we dont have any padding
        const size_t spitch = width * sizeof(uchar4);
        // Copy data located at address image_data in host memory to device memory
        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, image_data, spitch, width * sizeof(uchar4), height, cudaMemcpyHostToDevice));

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModePoint;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        // Create texture object
        //TODO: Texobj should be stored in device memory instead.
        cudaTextureObject_t texObj;
        checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

        PrintFirstPixel<<<1,1>>>(texObj);
    
        // Free device memory
        cudaFreeArray(cuArray); // We might not be able ot delete this in case it also removes all the underlying data from the textureobject.

        //This will return an invalid pointer, since texObj is just a local variable. I won't fix this, though, since there's not actually a way to keep the textrure
        //in memory after the call returns to python. This entire file is probably going to be deleted after I archive it in a commit. 
        return &texObj;
    }

 
    // Allocates a new array from the input array where every 4th index is a padded value of 1. The input pointer is overwritten with the pointer to the new array.
    // The array is 4/3rds the length of the input array. Remember to delete the allocated array.
    __global__ void CreatPaddedArrayFromBase(float* src, float* dest, int pixelDataCount){
        // Increase count by 1/3rd to make room for the 4th channel
        int paddedDataCount = pixelDataCount/3*4;

        for (size_t padded_i = 0; padded_i < paddedDataCount; padded_i++)
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
    }

    // Creates a textureObject wrapper around the provided texture data
    // Adapted from the lodepng decoding example example at https://github.com/lvandeve/lodepng/blob/master/examples/example_decode.cpp
    // and the cuda example at https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
    cudaTextureObject_t* CreateTexture(std::map<std::string, torch::Tensor> textureData){
        // extract all the texture data
        int height = textureData["height"].const_data_ptr<int>()[0];
        int width = textureData["width"].const_data_ptr<int>()[0];
        std::string mode = decodeTextureMode(textureData["mode"].const_data_ptr<int>()[0]);
        float* pixelData = textureData["pixelData"].contiguous().mutable_data_ptr<float>();
        int pixelDataCount = textureData["pixelData"].numel();
        std::cout << "Height & width: " << height << ", " << width << "; Mode: " << mode << "; pixelDataCount: " << pixelDataCount << std::endl;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));

        //TODO: allow the wrap mode to be set based on a texture import setting
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;

        //TODO: allow the coordinate mode ot be set based on a texture import setting
        texDesc.normalizedCoords = 1;

        cudaChannelFormatDesc channelDesc;
        int channelWidth;

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
        if(mode == "1" || mode == "L" || mode == "P" || mode == "I" || mode == "F"){
            channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            channelWidth = sizeof(float);
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
        }
        else if ( mode == "RGBA" || mode == "CMYK")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelWidth = sizeof(float) * 4;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;
        }

        else if ( mode == "RGB" || mode == "YCbCr")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
            channelWidth = sizeof(float) * 4;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeNormalizedFloat;

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3m so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            int paddedDataCount = pixelDataCount/3*4;
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataCount);
            CreatPaddedArrayFromBase<<<1,1>>>(pixelData, paddedData, pixelDataCount);
            // Overwrite the original data pointer. Remember to free the memory by the end of the function.s
            pixelData = paddedData;
        }
        else if (mode == "LAB" || mode == "HSV")
        {
            channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindUnsigned);
            channelWidth = sizeof(float) * 4;
            texDesc.filterMode = cudaFilterModePoint;  // Linear filtering is only available for floats 
            texDesc.readMode = cudaReadModeElementType;

            // CUDA only support textures with 1,2 or 4 channels pr. pixel, not 3m so we have to pad it with an additional value. In this case I'm just adding a 4th opaque alpha channel.
            int paddedDataCount = pixelDataCount/3*4;
            float* paddedData;
            cudaMalloc(&paddedData, paddedDataCount);
            CreatPaddedArrayFromBase<<<1,1>>>(pixelData, paddedData, pixelDataCount);
            // Overwrite the original data pointer. Remember to free the memory by the end of the function.
            pixelData = paddedData;
        }

        // Allocate CUDA array in device memory
        cudaArray_t cuArray;
        cudaMallocArray(&cuArray, &channelDesc, width, height);

        // Set pitch of the source (the width in memory in bytes of the 2D array pointed to by src)
        const size_t spitch = width * channelWidth;
        // Copy data located at address image_data in host memory to device memory
        // The reason we have to copy the data, and just supply a pointer to the contigious memory, is because the data is copied to a cudaArray,
        // which stores data in a space filling curve for better cache performance.

        std::cout << "Copying texture data to cudaArray" << std::endl;
        checkCudaErrors(cudaMemcpy2DToArray(cuArray, 0, 0, pixelData, spitch, width * channelWidth, height, cudaMemcpyDefault));
        std::cout << "Copy done" << std::endl;
        
        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;
       
        // Create texture object
        //TODO: Texobj should be stored in device memory instead.
        std::cout << "Creating textureObject" << std::endl;
        cudaTextureObject_t texObj;
        checkCudaErrors(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
        std::cout << "Object created. Trying to print first pixel value" << std::endl;
        
        PrintFirstPixel<<<1,1>>>(texObj);
    
        // Free device memory
        cudaFreeArray(cuArray); // We might not be able ot delete this in case it also removes all the underlying data from the textureobject.
        if ( mode == "RGB" || mode == "YCbCr" || mode == "LAB" || mode == "HSV")
        {
            cudaFree(pixelData);
        }
        std::cout << "K" << std::endl;
        // TODO: Make sure to keep track of which memory we need clean up at the end of this function, and at the end of this frame.
        return &texObj;
    }

    // TODO:unload texture
}



