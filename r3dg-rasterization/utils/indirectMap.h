#pragma once

#include <cuda.h>
#include <vector>
//#include "cuda_runtime.h"

// A non-optimized class for emulating a map interface on the GPU.
// Has one-time write acces on host, and read access on device.
template <typename KeyType, typename ValueType>
class IndirectMap
{
private:
    // Device stored variables:
    int *d_itemCount;
    KeyType* d_keys;
    ValueType* d_values;

    // Host stored variables:  asd
    // (None so far)
public:
    __host__ IndirectMap();

    // Upload data to GPU immediately upon instantiation.
    __host__ IndirectMap(std::vector<KeyType> keys, std::vector<ValueType> values);

    // Allocates and Uploads keys and values to internal device arrays.
    __host__ void SetAll(std::vector<KeyType> keys, std::vector<ValueType> values);

    //Read from map on device (Fast)
    __device__ ValueType Get(KeyType key);
    __device__ ValueType operator[](KeyType key);

    __host__ ~IndirectMap();
};

// A special case of the indirect map that uses char* keys and compares them as if they were strings.
template<class ValueType>
class IndirectMap<char*, ValueType>
{
private:
    // Device stored variables:
    int *d_itemCount;
    KeyType* d_keys;
    ValueType* d_values;

    // Host stored variables:  asd
    // (None so far)
public:
    __host__ IndirectMap();

    // Upload data to GPU immediately upon instantiation.
    __host__ IndirectMap(std::vector<char*> keys, std::vector<ValueType> values);

    // Allocates and Uploads keys and values to internal device arrays.
    __host__ void SetAll(std::vector<char*> keys, std::vector<ValueType> values);

    //Read from map on device (Fast). Compares null terminated char arrays
    __device__ ValueType Get(char* string);
    __device__ ValueType operator[](char* string);

    __host__ ~IndirectMap();
};