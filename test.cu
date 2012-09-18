#include <thrust/tuple.h>
#include <iostream>
#include "oet.h"
#include "bubble.h"
#include "transpose.h"
#include "memory.h"
#include <thrust/device_vector.h>

using namespace trove;

template<typename Value>
__global__ void test_transpose_indices(Value* r) {
    int global_index = threadIdx.x;
    Value warp_offsets;
    int rotation;
    compute_indices(warp_offsets, rotation);
    r[global_index] = warp_offsets;
}

template<int size, typename T>
__global__ void test_transpose(T* r) {
    typedef typename homogeneous_tuple<size, T>::type Value;
    typedef typename homogeneous_tuple<size, int>::type Indices;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Indices warp_offsets;
    int rotation;
    compute_indices(warp_offsets, rotation);

    Value data;
    data = counting_tuple<Value>::impl(
        global_index * size);
    
    for(int i = 0; i < 1; i++) {
        warp_transpose(data, warp_offsets, rotation);
    }
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_idx = threadIdx.x & WARP_MASK;
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    warp_store(data, warp_ptr, warp_idx, 32);
}





template<int size, typename T>
__global__ void test_uncoalesced_store(T* r) {
    
    typedef typename homogeneous_tuple<size, T>::type Value;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Value data = counting_tuple<Value>::impl(
        global_index * size);
    
    T* thread_ptr = r + global_index * size;
    uncoalesced_store(data, thread_ptr);
}



template<int size, typename T>
__global__ void test_shared_transpose(T* r) {
    typedef typename homogeneous_tuple<size, T>::type Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int work_per_thread = thrust::tuple_size<Value>::value;
    extern __shared__ T smem[];
    
    Value data;
    data = counting_tuple<Value>::impl(
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


template<typename Tuple>
struct tuple_sum_impl {};

template<typename HT, typename TT>
struct tuple_sum_impl<thrust::detail::cons<HT, TT> > {
    __host__ __device__ static HT
    impl(const thrust::detail::cons<HT, TT>& t, HT p) {
        return tuple_sum_impl<TT>::impl(t.get_tail(), p + t.get_head());
    }
};

template<>
struct tuple_sum_impl<thrust::null_type> {
    template<typename T>
    __host__ __device__ static T impl(thrust::null_type, const T& p) {
        return p;
    }
};

template<typename Tuple>
__host__ __device__
typename thrust::tuple_element<0, Tuple>::type tuple_sum(const Tuple& t) {
    return tuple_sum_impl<typename cons_type<Tuple>::type>::impl(t, 0);
}

template<typename Key, typename Value>
__global__ void test_oet(
    Key k, Value v,
    typename thrust::tuple_element<0, Value>::type* r) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    oet_sort_by_key(k, v);
    r[global_index] = tuple_sum(v);
}

template<typename Key, typename Value>
__global__ void test_bubble(
    Key k, Value v,
    typename thrust::tuple_element<0, Value>::type* r) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    bubble_sort_by_key(k, v);
    r[global_index] = tuple_sum(v);
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

#define ARITY 7

int main() {

    typedef typename homogeneous_tuple<5, int>::type five_int;
    // five_int c = counting_tuple<five_int>::impl(15);
    // print_tuple(c);
    
    // typedef thrust::tuple<int, int, int, int, int,
    //                       int, int, int, int, int> ten_int;
    // ten_int x = thrust::make_tuple(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    // thrust::device_vector<ten_int> d(1);
    // test<<<1,1>>>(x, 100, 7, thrust::raw_pointer_cast(d.data()));
    // ten_int y = d[0];
    // print_tuple(y);


    int n_blocks = 15 * 8 * 100;
    //int n_blocks = 1;
    int block_size = 256;

    thrust::device_vector<int> e(n_blocks*block_size*ARITY);

    // typedef typename homogeneous_tuple<5, int>::type n_int;
    // thrust::device_vector<n_int> f(32);
    // test_transpose_indices<<<1, 32>>>(
    //     thrust::raw_pointer_cast(f.data()));
    // for(int i = 0; i < 32; i++) {
    //     n_int z = f[i];
    //     std::cout << i << ": ";
    //     print_tuple(z);
    // }


    test_transpose<ARITY><<<n_blocks, block_size>>>(
        thrust::raw_pointer_cast(e.data()));
    verify(e);
    thrust::fill(e.begin(), e.end(), 0);
    test_uncoalesced_store<ARITY><<<n_blocks, block_size>>>(
        thrust::raw_pointer_cast(e.data()));
    verify(e);
    thrust::fill(e.begin(), e.end(), 0);
    test_shared_transpose<ARITY><<<n_blocks, block_size,
        sizeof(int) * ARITY * block_size>>>(
            thrust::raw_pointer_cast(e.data()));
    verify(e);
    thrust::fill(e.begin(), e.end(), 0);

    // five_int five_k = thrust::make_tuple(9,8,7,6,5);
    // five_int five_v = thrust::make_tuple(0,1,2,3,4);
    // five_int five_k_s = five_k;
    // five_int five_v_s = five_v;
    // oet_sort_by_key(five_k_s, five_v_s);
    // print_tuple(five_k_s);
    // print_tuple(five_v_s);
    // five_k_s = five_k;
    // five_v_s = five_v;
    // bubble_sort_by_key(five_k_s, five_v_s);
    // print_tuple(five_k_s);
    // print_tuple(five_v_s);
 
    // thrust::device_vector<int> r(n_blocks * block_size);
    // int iterations = 1;
    // cudaEvent_t start,stop;
    // float time=0;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start, 0);
    // for (int i = 0; i < iterations; i++) {
    //     test_oet<<<n_blocks, block_size>>>(five_k,
    //                                        five_v,
    //                                        thrust::raw_pointer_cast(r.data()));
    // }
    
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // time = time / (float)iterations;
    // std::cout << "  Time: " << time << " ms" << std::endl;
    // float kps = (float)n_blocks * block_size * 5  / (time*1000);
    // std::cout << "  Throughput: " << kps << " Mkeys/s" << std::endl
    //           << std::endl;

    // cudaEventRecord(start, 0);
    // for(int i = 0; i < iterations; i++) {
    //     test_bubble<<<n_blocks, block_size>>>(five_k,
    //                                           five_v,
    //                                           thrust::raw_pointer_cast(r.data()));
    // }
    
    // cudaEventRecord(stop, 0);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop);
    // time = time / (float)iterations;
    // std::cout << "  Time: " << time << " ms" << std::endl;
    // kps = (float)n_blocks * block_size * 5  / (time*1000);
    // std::cout << "  Throughput: " << kps << " Mkeys/s" << std::endl
    //           << std::endl;

    
}
    
