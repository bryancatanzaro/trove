#include <thrust/tuple.h>
#include <iostream>
#include "oet.h"
#include "bubble.h"

/*! \p cons_type is a metafunction that computes the
 * <tt>thrust::detail::cons</tt> type for a <tt>thrust::tuple</tt>
 */
template<typename Tuple>
struct cons_type {
    typedef thrust::detail::cons<
        typename Tuple::head_type,
        typename Tuple::tail_type> type;
};

template<int N, typename T>
struct homogeneous_tuple {
    typedef thrust::detail::cons<
        T, typename homogeneous_tuple<N-1, T>::type> type;
};

template<typename T>
struct homogeneous_tuple<0, T> {
    typedef thrust::null_type type;
};


// template<typename Tuple>
// struct update_tuple_impl{};

// template<typename HT, typename TT>
// struct update_tuple_impl<thrust::detail::cons<HT, TT> > {
//     __host__ __device__
//     static void impl(thrust::detail::cons<HT, TT>& tup, const HT& d, int idx) {
//         if(idx == 0) tup.get_head() = d;
//         update_tuple_impl<TT>::impl(tup.get_tail(), d, idx-1);
//     }
// };

// template<>
// struct update_tuple_impl<thrust::null_type> {
//     template<typename T>
//     __host__ __device__
//     static void impl(thrust::null_type, T, int) {}
// };

// template<typename Tuple, typename T>
// __host__ __device__
// void update_tuple(Tuple& tup, const T& d, int idx) {
//     update_tuple_impl<typename cons_type<Tuple>::type>::impl(tup, d, idx);
// }

#include <thrust/device_vector.h>

template<typename T>
struct print_tuple_helper{};

template<typename HT, typename TT>
struct print_tuple_helper<thrust::detail::cons<HT, TT> > {
    static void impl(const thrust::detail::cons<HT, TT>& a) {
        std::cout << a.get_head() << " ";
        print_tuple_helper<TT>::impl(a.get_tail());
    }
};

template<>
struct print_tuple_helper<thrust::null_type> {
    static void impl(thrust::null_type) {
        std::cout << std::endl;
    }
};

template<typename Tuple>
__host__
void print_tuple(const Tuple& a) {
    print_tuple_helper<typename cons_type<Tuple>::type>::impl(a);
}

template<typename Tuple>
__global__ void test(Tuple t, int d, int i, Tuple* r) {
    update_tuple(t, d, i);
    *r = t;
}


template<int s>
struct offset_constants{};

// This Python code computes the necessary magic constants for arbitrary sizes
// m: Number of elements per thread
// n: Number of threads per warp
//
// def offset(m, n):
//     for i in range(m):
//         val = n * i
//         if (n * i) % m == 1:
//             return val / m

// def permute(m, n):
//     o = offset(m, n)
//     return m-((n-1)/o+1)
        
template<>
struct offset_constants<5> {
    static const int offset=19;
    static const int permute=3;
};

template<>
struct offset_constants<7> {
    static const int offset=9;
    static const int permute=3;
};

template<>
struct offset_constants<9> {
    static const int offset=7;
    static const int permute=4;
};

#define WARP_SIZE 32
#define WARP_MASK 0x1f

template<typename IntTuple, int b, int o>
struct compute_offsets_impl{};

template<typename HT, typename TT, int b, int o>
struct compute_offsets_impl<thrust::detail::cons<HT, TT>, b, o> {
    typedef thrust::detail::cons<HT, TT> Tuple;
    __device__
    static Tuple impl(int offset) {
        if (offset >= b) {
            offset -= b;
        } //Poor man's x % b. Requires that o < b.
        return Tuple(offset,
                     compute_offsets_impl<TT, b, o>::
                     impl(offset + o));
    }
};

template<int b, int o>
struct compute_offsets_impl<thrust::null_type, b, o> {
    __device__
    static thrust::null_type impl(int) {
        return thrust::null_type();
    }
};

template<int m>
__device__
typename homogeneous_tuple<m, int>::type compute_offsets() {
    typedef offset_constants<m> constants;
    typedef typename homogeneous_tuple<m, int>::type result_type;
    int warp_id = threadIdx.x & WARP_MASK;
    int initial_offset = ((WARP_SIZE - warp_id) * constants::offset)
        & WARP_MASK;
    return compute_offsets_impl<result_type,
                                WARP_SIZE,
                                constants::offset>::impl(initial_offset);
}
        
template<int m>
__device__
typename homogeneous_tuple<m, int>::type compute_permute() {
    typedef offset_constants<m> constants;
    typedef typename homogeneous_tuple<m, int>::type result_type;
    int warp_id = threadIdx.x & WARP_MASK;
    int initial_offset = ((m - constants::permute) * warp_id) % m;
    return compute_offsets_impl<result_type,
                                m, constants::permute>::impl(initial_offset);
}

template<typename T>
struct counting_tuple{};

template<typename HT, typename TT>
struct counting_tuple<thrust::detail::cons<HT, TT> > {
    typedef thrust::detail::cons<HT, TT> Tuple;
    __host__ __device__
    static Tuple impl(HT v=0, HT i=1) {
        return Tuple(v,
                     counting_tuple<TT>::impl(v + i));
    }
};

template<>
struct counting_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__
    static thrust::null_type impl(T v=0, T i=1) {
        return thrust::null_type();
    }
};

template<typename Data, typename Indices>
struct transpose_warp_tuples {};

template<typename DHT, typename DTT, typename IHT, typename ITT>
struct transpose_warp_tuples<
    thrust::detail::cons<DHT, DTT>,
    thrust::detail::cons<IHT, ITT> > {
    __device__ static void impl(thrust::detail::cons<DHT, DTT>& d,
                                const thrust::detail::cons<IHT, ITT>& i) {
        d.get_head() = __shfl(d.get_head(), i.get_head());
        transpose_warp_tuples<DTT, ITT>::impl(d.get_tail(),
                                              i.get_tail());
    }
};

template<>
struct transpose_warp_tuples<
    thrust::null_type, thrust::null_type> {
    __device__ static void impl(thrust::null_type, thrust::null_type) {}
};

template<typename Tuple>
struct store_tuple {};

template<typename HT, typename TT>
struct store_tuple<thrust::detail::cons<HT, TT> > {
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        HT* ptr, int offset, int stride) {
        ptr[offset] = d.get_head();
        store_tuple<TT>::impl(d.get_tail(), ptr, offset + stride, stride);
    }
};

template<>
struct store_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, T*, int, int) {}
};

template<typename Value>
__global__ void test_transpose_indices(Value* r) {
    int global_index = threadIdx.x;
    Value warp_offsets = compute_offsets<thrust::tuple_size<Value>::value>();
    Value tuple_indices = compute_permute<thrust::tuple_size<Value>::value>();
    r[global_index] = tuple_indices;
}

template<int size, typename T>
__global__ void test_transpose(T* r) {
    typedef typename homogeneous_tuple<size, T>::type Value;
    
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    Value warp_offsets = compute_offsets<thrust::tuple_size<Value>::value>();
    Value permutation = compute_permute<thrust::tuple_size<Value>::value>();
    
    Value data = counting_tuple<Value>::impl(
       global_index * size);

    transpose_warp_tuples<Value, Value>::impl(data, warp_offsets);
    oet_sort_by_key(permutation, data);
    int warp_begin = threadIdx.x & (~WARP_MASK);
    int warp_idx = threadIdx.x & WARP_MASK;
    int warp_offset = (blockDim.x * blockIdx.x + warp_begin) * size;
    T* warp_ptr = r + warp_offset;
    store_tuple<Value>::impl(data, warp_ptr, warp_idx, 32);
}


template<typename T>
struct uncoalesced_store_tuple{};

template<typename HT, typename TT>
struct uncoalesced_store_tuple<thrust::detail::cons<HT, TT> > {
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        HT* ptr,
        int offset=0) {
        ptr[offset] = d.get_head();
        uncoalesced_store_tuple<TT>::impl(d.get_tail(), ptr, offset+1);
    }
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        volatile HT* ptr,
        int offset=0) {
        ptr[offset] = d.get_head();
        uncoalesced_store_tuple<TT>::impl(d.get_tail(), ptr, offset+1);
    }
};

template<>
struct uncoalesced_store_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, T*, int) {}
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, volatile T*, int) {}
};

template<int size, typename T>
__global__ void test_uncoalesced_write(T* r) {
    
    typedef typename homogeneous_tuple<size, T>::type Value;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;

    Value data = counting_tuple<Value>::impl(
        global_index * size);
    
    T* thread_ptr = r + global_index * size;
    uncoalesced_store_tuple<Value>::impl(data, thread_ptr);
}

template<int size, typename T>
__global__ void test_shared_transpose(T* r) {
    typedef typename homogeneous_tuple<size, T>::type Value;

    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int work_per_thread = thrust::tuple_size<Value>::value;
    extern __shared__ T smem[];
    
    Value data = counting_tuple<Value>::impl(
        global_index * work_per_thread);
    
    T* thread_ptr = smem + threadIdx.x * work_per_thread;
    uncoalesced_store_tuple<Value>::impl(data, thread_ptr);
    __syncthreads();
    T* block_ptr = r + blockDim.x * blockIdx.x * work_per_thread;
    for(int i = threadIdx.x; i < work_per_thread * blockDim.x; i += blockDim.x) {
        block_ptr[i] = smem[i];
    }
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
    int block_size = 256;

    thrust::device_vector<int> e(n_blocks*block_size*5);

    // typedef typename homogeneous_tuple<5, int>::type n_int;
    // thrust::device_vector<n_int> f(32);
    // test_transpose_indices<<<1, 32>>>(
    //     thrust::raw_pointer_cast(f.data()));
    // for(int i = 0; i < 32; i++) {
    //     n_int z = f[i];
    //     std::cout << i << ": ";
    //     print_tuple(z);
    // }


    test_transpose<5><<<n_blocks, block_size>>>(
        thrust::raw_pointer_cast(e.data()));
    verify(e);
    test_uncoalesced_write<5><<<n_blocks, block_size>>>(
        thrust::raw_pointer_cast(e.data()));
    verify(e);
    test_shared_transpose<5><<<n_blocks, block_size,
        sizeof(int) * 5 * block_size>>>(
            thrust::raw_pointer_cast(e.data()));
    verify(e);
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
    
