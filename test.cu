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

template<typename Tuple>
struct update_tuple_impl{};

template<typename HT, typename TT>
struct update_tuple_impl<thrust::detail::cons<HT, TT> > {
    __host__ __device__
    static void impl(thrust::detail::cons<HT, TT>& tup, const HT& d, int idx) {
        if(idx == 0) tup.get_head() = d;
        update_tuple_impl<TT>::impl(tup.get_tail(), d, idx-1);
    }
};

template<>
struct update_tuple_impl<thrust::null_type> {
    template<typename T>
    __host__ __device__
    static void impl(thrust::null_type, T, int) {}
};

template<typename Tuple, typename T>
__host__ __device__
void update_tuple(Tuple& tup, const T& d, int idx) {
    update_tuple_impl<typename cons_type<Tuple>::type>::impl(tup, d, idx);
}

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


template<typename IntTuple, int m>
struct compute_offsets_impl{};

template<typename HT, typename TT, int m>
struct compute_offsets_impl<thrust::detail::cons<HT, TT>, m> {
    typedef thrust::detail::cons<HT, TT> IntTuple;
    __host__ __device__ static void impl(unsigned short idx, IntTuple& k, IntTuple& v) {
        k.get_head() = idx % m;
        v.get_head() = idx / m;
        compute_offsets_impl<TT, m>::impl(idx + 32, k.get_tail(), v.get_tail());
    }
};

template<int m>
struct compute_offsets_impl<thrust::null_type, m> {
    __host__ __device__ static void impl(unsigned short, thrust::null_type, thrust::null_type) {}
};

template<typename IntTuple>
__device__ void compute_offsets(IntTuple& k, IntTuple& v) {
    unsigned short warp_idx = threadIdx.x & 0x1f;
    compute_offsets_impl<typename cons_type<IntTuple>::type,
                         thrust::tuple_size<IntTuple>::value>::impl(warp_idx, k, v);
}


template<typename Value>
__global__ void test_sort(Value* r) {

    int global_index = threadIdx.x;
    Value k, v;
    compute_offsets(k, v);
    
    oet_sort_by_key(k, v);
    r[global_index] = v;
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

int main() {
    typedef thrust::tuple<int, int, int, int, int,
                          int, int, int, int, int> ten_int;
    ten_int x = thrust::make_tuple(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
    thrust::device_vector<ten_int> d(1);
    test<<<1,1>>>(x, 100, 7, thrust::raw_pointer_cast(d.data()));
    ten_int y = d[0];
    print_tuple(y);


    
    typedef thrust::tuple<int, int, int, int, int> five_int;
    thrust::device_vector<five_int> e(32);
    test_sort<<<1,32>>>(thrust::raw_pointer_cast(e.data()));
    for(int i = 0; i < 32; i++) {
        five_int z = e[i];
        std::cout << i << ": ";
        print_tuple(z);
    }

    five_int five_k(9,8,7,6,5);
    five_int five_v(0,1,2,3,4);
    five_int five_k_s = five_k;
    five_int five_v_s = five_v;
    oet_sort_by_key(five_k_s, five_v_s);
    print_tuple(five_k_s);
    print_tuple(five_v_s);
    five_k_s = five_k;
    five_v_s = five_v;
    bubble_sort_by_key(five_k_s, five_v_s);
    print_tuple(five_k_s);
    print_tuple(five_v_s);
 
    int n_blocks = 15 * 8 * 100;
    int block_size = 256;
    thrust::device_vector<int> r(n_blocks * block_size);
    int iterations = 1;
    cudaEvent_t start,stop;
    float time=0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < iterations; i++) {
        test_oet<<<n_blocks, block_size>>>(five_k,
                                           five_v,
                                           thrust::raw_pointer_cast(r.data()));
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time = time / (float)iterations;
    std::cout << "  Time: " << time << " ms" << std::endl;
    float kps = (float)n_blocks * block_size * 5  / (time*1000);
    std::cout << "  Throughput: " << kps << " Mkeys/s" << std::endl
              << std::endl;

    cudaEventRecord(start, 0);
    for(int i = 0; i < iterations; i++) {
        test_bubble<<<n_blocks, block_size>>>(five_k,
                                              five_v,
                                              thrust::raw_pointer_cast(r.data()));
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    time = time / (float)iterations;
    std::cout << "  Time: " << time << " ms" << std::endl;
    kps = (float)n_blocks * block_size * 5  / (time*1000);
    std::cout << "  Throughput: " << kps << " Mkeys/s" << std::endl
              << std::endl;

    
}
    
