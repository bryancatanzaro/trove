/*
Copyright (c) 2013, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <cstdlib>

#include <thrust/device_vector.h>
#include <thrust/equal.h>



#include <trove/ptr.h>
#include <trove/aos.h>
#include "timer.h"

using namespace trove;

template<typename T>
__global__ void
benchmark_contiguous_shfl_store(T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    T data;
    int size = detail::aliased_size<T, int>::value;
    data = counting_array<T>::impl(
        global_index * size);
    store_warp_contiguous(data, r + global_index);    
}

template<typename T>
__global__ void
benchmark_contiguous_direct_store(T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    T data;
    int size = detail::aliased_size<T, int>::value;
    data = counting_array<T>::impl(
        global_index * size);
    r[global_index] = data;
}

template<typename T>
__global__ void
benchmark_contiguous_shfl_load(T* s, typename T::value_type* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    T data = load_warp_contiguous(s + global_index);
    r[global_index] = sum(data);
}

template<typename T>
__global__ void
benchmark_contiguous_direct_load(T* s, typename T::value_type* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    T data = s[global_index];
    r[global_index] = sum(data);
}

template<typename T>
__global__ void
benchmark_shfl_gather(const int* indices, T* raw_s, T* raw_r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    trove::coalesced_ptr<T> s(raw_s);
    trove::coalesced_ptr<T> r(raw_r);
    T data = s[index];
    r[global_index] = data;
}

template<typename T>
__global__ void
benchmark_shfl_scatter(const int* indices, T* raw_s, T* raw_r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    trove::coalesced_ptr<T> s(raw_s);
    trove::coalesced_ptr<T> r(raw_r);
    T data = s[global_index];
    r[index] = data;
}

template<typename T>
__global__ void
benchmark_direct_gather(const int* indices, T* s, T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    T data = s[index];
    r[global_index] = data;
}

template<typename T>
__global__ void
benchmark_direct_scatter(const int* indices, T* s, T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    T data = s[global_index];
    r[index] = data;
}

template<int i>
void run_benchmark_contiguous_store(const std::string name, void (*test)(array<int, i>*),
                                    void (*gold)(array<int, i>*)) {
    typedef array<int, i> T;

    std::cout << name << ", " << i << ", ";
    int n_blocks = 80 * 8 * 100;
    int block_size = 256;
    int n = n_blocks * block_size - 100;
    thrust::device_vector<T> r(n);
    int iterations = 10;
    cuda_timer timer;
    timer.start();
    for(int j = 0; j < iterations; j++) {
        test<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(r.data()));
    }
    float time = timer.stop();
    float gbs = (float)(sizeof(T) * (iterations * n_blocks * block_size)) / (time * 1000000);
    std::cout << gbs << ", ";
    bool correct = true;
    if (test != gold) {
        thrust::device_vector<T> g(n);
        gold<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(g.data()));
        correct = thrust::equal(r.begin(), r.end(), g.begin());
    }
    if (correct)
        std::cout << "Results passed";
    else
        std::cout << "INCORRECT";
    std::cout << std::endl;
    
}

template<int i>
struct run_benchmark_contiguous_shfl_store {
    typedef array<int, i> T;
    static void impl() {
        run_benchmark_contiguous_store("Contiguous SHFL Store", &benchmark_contiguous_shfl_store<T>,
                                       &benchmark_contiguous_direct_store<T>);
    }
};

template<int i>
struct run_benchmark_contiguous_direct_store {
    typedef array<int, i> T;
    static void impl() {
        run_benchmark_contiguous_store("Contiguous Direct Store", &benchmark_contiguous_direct_store<T>,
                                       &benchmark_contiguous_direct_store<T>);
    }
};



template<typename T>
void fill_test(thrust::device_vector<T>& d) {
    thrust::device_ptr<int> p = thrust::device_ptr<int>((int*)thrust::raw_pointer_cast(d.data()));
    thrust::counting_iterator<int> c(0);
    int s = d.size() * sizeof(T) / sizeof(int);
    thrust::copy(c, c+s, p);
}

template<int i>
void run_benchmark_contiguous_load(const std::string name, void (*test)(array<int, i>*, int*),
                                         void (*gold)(array<int, i>*, int*)) {
    typedef array<int, i> T;

    std::cout << name << ", " << i << ", ";
    int n_blocks = 80 * 8 * 100;
    int block_size = 256;
    int n = n_blocks * block_size;
    thrust::device_vector<T> s(n);
    fill_test(s);
    thrust::device_vector<int> r(n);
    int iterations = 10;
    cuda_timer timer;
    timer.start();
    for(int j = 0; j < iterations; j++) {
        test<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(s.data()), thrust::raw_pointer_cast(r.data()));
    }
    float time = timer.stop();
    float gbs = (float)((sizeof(T) + sizeof(int)) * (iterations * n_blocks * block_size)) / (time * 1000000);
    std::cout << gbs << ", ";
    bool correct = true;
    if (test != gold) {
        thrust::device_vector<int> g(n);
        gold<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(s.data()), thrust::raw_pointer_cast(g.data()));
        correct = thrust::equal(r.begin(), r.end(), g.begin());
    }
    
    if (correct)
        std::cout << "Results passed";
    else
        std::cout << "INCORRECT";
    std::cout << std::endl;
            
    
}

template<int i>
struct run_benchmark_contiguous_shfl_load {
    typedef array<int, i> T;
    static void impl() {
        run_benchmark_contiguous_load("Contiguous SHFL Load", &benchmark_contiguous_shfl_load<T>, &benchmark_contiguous_direct_load<T>);
    }
};

template<int i>
struct run_benchmark_contiguous_direct_load {
    typedef array<int, i> T;
    static void impl() {
        run_benchmark_contiguous_load("Contiguous Direct Load", &benchmark_contiguous_direct_load<T>, &benchmark_contiguous_direct_load<T>);
    }
};

thrust::device_vector<int> make_device_random(int s) {
    thrust::host_vector<int> h(s);
    thrust::generate(h.begin(), h.end(), rand);
    thrust::device_vector<int> d = h;
    return d;
}

thrust::device_vector<int> make_random_permutation(int s) {
    thrust::device_vector<int> keys = make_device_random(s);
    thrust::counting_iterator<int> c(0);
    thrust::device_vector<int> values(s);
    thrust::copy(c, c+s, values.begin());
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    return values;
}

template<int i>
void run_benchmark_random(const std::string name, const thrust::device_vector<int>& permutation,
                          void (*test)(const int*, array<int, i>*, array<int, i>*),
                          void (*gold)(const int*, array<int, i>*, array<int, i>*)) {
    typedef array<int, i> T;

    std::cout << name << ", " << i << ", ";
    int n_blocks = 80 * 8 * 100;
    int block_size = 256;
    int n = n_blocks * block_size;
    thrust::device_vector<T> s(n);
    fill_test(s);
    thrust::device_vector<T> r(n);
    int iterations = 10;
    cuda_timer timer;
    timer.start();
    for(int j = 0; j < iterations; j++) {
        test<<<n_blocks, block_size>>>(
            thrust::raw_pointer_cast(permutation.data()),
            thrust::raw_pointer_cast(s.data()),
            thrust::raw_pointer_cast(r.data()));
    }
    float time = timer.stop();
    float gbs = (float)(sizeof(T) * (2 * iterations * n_blocks * block_size) + sizeof(int) * iterations * n_blocks * block_size) / (time * 1000000);
    std::cout << gbs << ", ";
    bool correct = true;
    if (test != gold) {
        thrust::device_vector<T> g(n);
        gold<<<n_blocks, block_size>>>(thrust::raw_pointer_cast(permutation.data()),
                                       thrust::raw_pointer_cast(s.data()), thrust::raw_pointer_cast(g.data()));
        correct = thrust::equal(r.begin(), r.end(), g.begin());
    }
    if (correct)
        std::cout << "Results passed";
    else
        std::cout << "INCORRECT";
    std::cout << std::endl;
}

template<int i>
struct run_benchmark_shfl_gather {
    typedef array<int, i> T;
    static void impl(const thrust::device_vector<int>& permutation) {
        run_benchmark_random("SHFL Gather", permutation, &benchmark_shfl_gather<T>, &benchmark_direct_gather<T>);
    }
};

template<int i>
struct run_benchmark_direct_gather {
    typedef array<int, i> T;
    static void impl(const thrust::device_vector<int>& permutation) {
        run_benchmark_random("Direct Gather", permutation, &benchmark_direct_gather<T>, &benchmark_direct_gather<T>);
    }
};

template<int i>
struct run_benchmark_shfl_scatter {
    typedef array<int, i> T;
    static void impl(const thrust::device_vector<int>& permutation) {
        run_benchmark_random("SHFL Scatter", permutation, &benchmark_shfl_scatter<T>, &benchmark_direct_scatter<T>);
    }
};
template<int i>
struct run_benchmark_direct_scatter {
    typedef array<int, i> T;
    static void impl(const thrust::device_vector<int>& permutation) {
        run_benchmark_random("Direct Scatter", permutation, &benchmark_direct_scatter<T>, &benchmark_direct_scatter<T>);
    }
};

template<template<int> class F, typename Cons>
struct do_tests {
    static void impl() {
        F<Cons::head>::impl();
        do_tests<F, typename Cons::tail>::impl();
    }
    template<typename T>
    static void impl(const T& t) {
        F<Cons::head>::impl(t);
        do_tests<F, typename Cons::tail>::impl(t);
    }
};

template<template<int> class F>
struct do_tests<F, null_type> {
    static void impl() {}
    template<typename T>
    static void impl(const T& t) {}
};

#ifndef LOWER_BOUND
#define LOWER_BOUND 1
#endif
#ifndef UPPER_BOUND
#define UPPER_BOUND 16
#endif

typedef static_range<LOWER_BOUND, UPPER_BOUND> sizes;

int main() {
    do_tests<run_benchmark_contiguous_shfl_store, sizes>::impl();
    do_tests<run_benchmark_contiguous_direct_store, sizes>::impl();
    do_tests<run_benchmark_contiguous_shfl_load, sizes>::impl();
    do_tests<run_benchmark_contiguous_direct_load, sizes>::impl();
    int size = 80 * 8 * 100 * 256;
    thrust::device_vector<int> permutation = make_random_permutation(size);
    do_tests<run_benchmark_shfl_scatter, sizes>::impl(permutation);
    do_tests<run_benchmark_direct_scatter, sizes>::impl(permutation);
    do_tests<run_benchmark_shfl_gather, sizes>::impl(permutation);
    do_tests<run_benchmark_direct_gather, sizes>::impl(permutation);

}

