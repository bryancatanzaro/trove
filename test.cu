#include <iostream>
#include "transpose.h"
#include "memory.h"
#include "print_array.h"
#include <thrust/device_vector.h>

using namespace trove;

template<int size, typename T>
__global__ void test_c2r_transpose(T* r) {
    typedef array<T, size> Value;
    typedef array<int, size> Indices;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Indices warp_offsets;
    int rotation;
    c2r_compute_indices(warp_offsets, rotation);

    Value data;
    data = counting_array<Value>::impl(
        global_index * size);
    
    c2r_warp_transpose(data, warp_offsets, rotation);

    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_idx = threadIdx.x & WARP_MASK;
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
}


template<int size, typename T>
__global__ void test_r2c_transpose(T* r) {
    typedef array<T, size> Value;
    typedef array<int, size> Indices;
  
    int global_warp_id = (threadIdx.x >> LOG_WARP_SIZE) + (blockDim.x >> LOG_WARP_SIZE) * blockIdx.x;
    int warp_idx = threadIdx.x & WARP_MASK;
    int start_value = ((global_warp_id * size) << LOG_WARP_SIZE) + warp_idx;
    
    Indices warp_offsets;
    int rotation;
    r2c_compute_indices(warp_offsets, rotation);

    Value data;
    data = counting_array<Value>::impl(
        start_value, WARP_SIZE);
    
    r2c_warp_transpose(data, warp_offsets, rotation);

    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
}




template<int size, typename T>
__global__ void test_uncoalesced_store(T* r) {
    
    typedef array<T, size> Value;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Value data = counting_array<Value>::impl(
        global_index * size);
    
    T* thread_ptr = r + global_index * size;
    uncoalesced_store(data, thread_ptr);
}



template<int size, typename T>
__global__ void test_shared_c2r_transpose(T* r) {
    typedef array<T, size> Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int work_per_thread = thrust::tuple_size<Value>::value;
    extern __shared__ T smem[];
    
    Value data;
    data = counting_array<Value>::impl(
        global_index * work_per_thread);
    int warp_id = threadIdx.x >> 5;
    int warp_idx = threadIdx.x & WARP_MASK;

    for(int i = 0; i < 1; i++) {
        volatile T* thread_ptr = smem + threadIdx.x * work_per_thread;
        uncoalesced_store(data, thread_ptr);


        data = warp_load<Value>(smem + warp_id * WARP_SIZE * size,
                                warp_idx);
    }
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
   
}



template<typename T>
void verify_c2r(thrust::device_vector<T>& d_r) {
    thrust::host_vector<T> h_r = d_r;
    bool fail = false;
    for(int i = 0; i < h_r.size(); i++) {
        if (h_r[i] != i) {
            std::cout << "  Fail: r[" << i << "] is " << h_r[i] << std::endl;
            fail = true;
        }
    }
    if (!fail) {
        std::cout << "Pass!" << std::endl;
    }
}

template<int size, typename T>
void verify_r2c(thrust::device_vector<T>& d_r) {
    thrust::host_vector<T> h_r = d_r;
    bool fail = false;

    int expected = 0;
    int warp_index = 0;
    int row_index = 0;
    for(int i = 0; i < h_r.size(); i++) {
        if (h_r[i] != expected) {
            std::cout << "  Fail: r[" << i << "] is " << h_r[i]
                      << " (expected " << expected << ")" << std::endl;
            fail = true;
        }
        expected += size;
        warp_index++;
        if (warp_index == 32) {
            expected -= 32 * size - 1;
            warp_index = 0;
            row_index++;
            if (row_index == size) {
                row_index = 0;
                expected += 31 * size;
            }
        }
    }
    if (!fail) {
        std::cout << "Pass!" << std::endl;
    }
}

struct null_type{};

template<typename T, T i, typename Tail=null_type>
struct cons_c {
    static const T head = i;
    typedef Tail tail;
};

typedef
cons_c<int, 2,
     cons_c<int, 3,
          cons_c<int, 4,
               cons_c<int, 5,
                    cons_c<int, 7,
                         cons_c<int, 8,
                              cons_c<int, 9,
                                   cons_c<int, 11,
                                        cons_c<int, 13,
                                             cons_c<int, 15,
                                                  cons_c<int, 16,
                                                       null_type
                                                       > > > > > > > > > > > c2r_arities;

typedef
cons_c<int, 2,
     cons_c<int, 3,
          cons_c<int, 4,
               cons_c<int, 5,
                    cons_c<int, 7,
                         cons_c<int, 8,
                              cons_c<int, 9,
                                   cons_c<int, 11,
                                        cons_c<int, 13,
                                             cons_c<int, 15,
                                                  cons_c<int, 16,
                                                       null_type
                                                       > > > > > > > > > > > r2c_arities;

template<typename T, int i>
void print_warp_result(const thrust::device_vector<T> e) {
    thrust::host_vector<int> f = e;
    for(int row = 0; row < i; row++) {
        for(int col = 0; col < 32; col++) {
            int r = f[col + row * 32];
            if (r < 10) std::cout << " ";
            if (r < 100) std::cout << " ";
            std::cout << f[col + row * 32] << " ";
        }
        std::cout << std::endl;
    }
}


template<int i>
struct test_c2r {
    static void impl() {
        std::cout << "Testing c2r transpose for " <<
            i << " elements per thread" << std::endl;
        int n_blocks = 15 * 8 * 100;
        int block_size = 256;
        thrust::device_vector<int> e(n_blocks*block_size*i);
        test_c2r_transpose<i>
            <<<n_blocks, block_size>>>(thrust::raw_pointer_cast(e.data()));
        verify_c2r(e);
    }
};

template<int i>
struct test_r2c {
    static void impl() {
        std::cout << "Testing r2c transpose for " <<
            i << " elements per thread" << std::endl;

        int n_blocks = 15 * 8 * 100;
        int block_size = 256;
        thrust::device_vector<int> e(n_blocks*block_size*i);
        test_r2c_transpose<i>
            <<<n_blocks, block_size>>>(thrust::raw_pointer_cast(e.data()));
        verify_r2c<i>(e);

    }
};

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
  
int main() {
    do_tests<test_c2r, c2r_arities>::impl();
    do_tests<test_r2c, r2c_arities>::impl();
 
}
    
