#include <thrust/tuple.h>
#include <iostream>
#include <trove/utility.h>
#include <trove/transpose.h>
#include <trove/memory.h>
#include <thrust/device_vector.h>
#include "timer.h"

using namespace trove;


template<int size, typename T>
__global__ void test_transpose(T* r) {
    typedef array<T, size> Value;
    typedef array<int, size> Indices;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Indices warp_offsets;
    int rotation;
    c2r_compute_indices(warp_offsets, rotation);

    Value data;
    data = counting_array<Value>::impl(
        global_index * size);
    
    for(int i = 0; i < 4096; i++) {
        c2r_warp_transpose(data, warp_offsets, rotation);
    }
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_idx = threadIdx.x & WARP_MASK;
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
}


template<int size, typename T>
__global__ void test_shared_transpose(T* r) {
    typedef array<T, size> Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ T smem[];
    
    Value data;
    data = counting_array<Value>::impl(
        global_index * size);
    int warp_id = threadIdx.x >> 5;
    int warp_idx = threadIdx.x & WARP_MASK;

    for(int i = 0; i < 4096; i++) {
        volatile T* thread_ptr = smem + threadIdx.x * size;
        uncoalesced_store(data, thread_ptr);

        __syncthreads();
        data = warp_load<Value>(smem + warp_id * WARP_SIZE * size,
                                warp_idx);
        __syncthreads();
    }
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
   
}

template<int size, typename T>
__global__ void test_unsafe_shared_transpose(T* r) {
    typedef array<T, size> Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ T smem[];
    
    Value data;
    data = counting_array<Value>::impl(
        global_index * size);
    int warp_id = threadIdx.x >> 5;
    int warp_idx = threadIdx.x & WARP_MASK;

    for(int i = 0; i < 4096; i++) {
        volatile T* thread_ptr = smem + threadIdx.x * size;
        uncoalesced_store(data, thread_ptr);


        data = warp_load<Value>(smem + warp_id * WARP_SIZE * size,
                                warp_idx);
    }
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
   
}



template<template<int> class F, typename Cons>
struct do_tests {
    static void impl() {
        F<Cons::head>::impl();
        do_tests<F, typename Cons::tail>::impl();
    }

};

template<template<int> class F>
struct do_tests<F, null_type> {
    static void impl() {}
};

template<int i>
struct run_shfl_transpose {
    static void impl() {
        
        int n_blocks = 13 * 8;
        int block_size = 256;

        thrust::device_vector<int> e(n_blocks*block_size*i);

        int max_iter = 10;
        cuda_timer t;
        t.start();
        for(int j = 0; j < max_iter; j++) {
            test_transpose<i><<<n_blocks, block_size>>>(
                thrust::raw_pointer_cast(e.data()));
        }
        float ms = t.stop();
        std::cout << "SHFL, " << i << ",  " << ms/max_iter << std::endl;
    }
};


template<int i>
struct run_unsafe_transpose {
    static void impl() {
        
        int n_blocks = 13 * 8;
        int block_size = 256;

        thrust::device_vector<int> e(n_blocks*block_size*i);

        cuda_timer t;
        t.start();
        int max_iter = 10;
        for(int j = 0; j < max_iter; j++) {
            test_unsafe_shared_transpose<i><<<n_blocks, block_size,
                sizeof(int) * i * block_size>>>(
                    thrust::raw_pointer_cast(e.data()));
        }
        float ms = t.stop();
        std::cout << "Unsafe, " << i << ",  " << ms/max_iter << std::endl;
    }
};

template<int i>
struct run_safe_transpose {
    static void impl() {
        
        int n_blocks = 13 * 8;
        int block_size = 256;

        thrust::device_vector<int> e(n_blocks*block_size*i);
        int max_iter = 10;
        cuda_timer t;
        t.start();
        for(int j = 0; j < max_iter; j++) {
            test_shared_transpose<i><<<n_blocks, block_size,
                sizeof(int) * i * block_size>>>(
                    thrust::raw_pointer_cast(e.data()));
        }
        float ms = t.stop();
        std::cout << "Safe, " << i << ",  " << ms/max_iter << std::endl;
    }
};

#ifndef LOWER_BOUND
#define LOWER_BOUND 1
#endif
#ifndef UPPER_BOUND
#define UPPER_BOUND 32
#endif

typedef static_range<LOWER_BOUND, UPPER_BOUND> sizes;


int main() {

    do_tests<run_shfl_transpose, sizes>::impl();
    do_tests<run_unsafe_transpose, sizes>::impl();
    do_tests<run_safe_transpose, sizes>::impl();
    
    
}
    
