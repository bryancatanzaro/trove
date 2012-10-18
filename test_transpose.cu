#include <iostream>
#include "transpose.h"
#include "aos.h"
#include "print_array.h"

#include <thrust/device_vector.h>



using namespace trove;

template<int size, typename T>
__global__ void test_c2r_transpose(T* r) {
    typedef array<T, size> Value;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Value data;
    data = counting_array<Value>::impl(
        global_index * size);
    store_aos_warp_contiguous(data, (array<T, size>*)r, global_index);

}


template<int size, typename T>
__global__ void test_r2c_transpose(T* s, T* r) {
    typedef array<T, size> Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    Value data = load_aos_warp_contiguous((array<T, size>*)s, global_index);
    store_aos_warp_contiguous(data, (array<T, size>*)r, global_index);
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
    warp_store(data, warp_ptr, warp_idx);
   
}



template<typename T>
void verify(thrust::device_vector<T>& d_r) {
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

typedef static_range<2, 16> c2r_arities;

typedef static_range<2, 16> r2c_arities;

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
        verify(e);
    }
};

template<int i>
struct test_r2c {
    static void impl() {
        std::cout << "Testing r2c transpose for " <<
            i << " elements per thread" << std::endl;
        int n_blocks = 15 * 8 * 100;
        int block_size = 256;
        thrust::device_vector<int> s(n_blocks*block_size*i);
        thrust::counting_iterator<int> begin(0);
        thrust::counting_iterator<int> end = begin + n_blocks*block_size*i;
        thrust::copy(begin, end, s.begin());
        thrust::device_vector<int> d(n_blocks*block_size*i);
        test_r2c_transpose<i>
            <<<n_blocks, block_size>>>(
                thrust::raw_pointer_cast(s.data()),
                thrust::raw_pointer_cast(d.data()));
        verify(d);
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
    
