#pragma once
#include "utility.h"
#include "rotate.h"
#include "shfl.h"

#define WARP_SIZE 32
#define WARP_MASK 0x1f
#define LOG_WARP_SIZE 5

namespace trove {
namespace detail {

template<int s>
struct c2r_offset_constants{};

// This Python code computes the necessary magic constants for
// arbitrary odd sizes
// m: Number of elements per thread
// n: Number of threads per warp
//

//def offset(m, n):
//    for i in range(m):
//        val = n * i
//        if val % m == 1:
//            return val / m
//
//def permute(m, n):
//    o = offset(m, n)
//    return (n-1)/o+1
//
//def rotate(m, n):
//    for i in range(m):
//        val = n * i
//        if val % m == 1:
//            return val / n

template<>
struct c2r_offset_constants<3> {
    static const int offset=21;
    static const int permute=2;
    static const int rotate=2;
};

template<>
struct c2r_offset_constants<5> {
    static const int offset=19;
    static const int permute=2;
    static const int rotate=3;
};

template<>
struct c2r_offset_constants<7> {
    static const int offset=9;
    static const int permute=4;
    static const int rotate=2;
};

template<>
struct c2r_offset_constants<9> {
    static const int offset=7;
    static const int permute=5;
    static const int rotate=2;
};

template<int m>
struct c2r_power_of_two_constants {
    static const int offset = WARP_SIZE - WARP_SIZE/m; 
    static const int permute = m - 1;
};

template<>
struct c2r_offset_constants<2> {
    static const int offset = c2r_power_of_two_constants<2>::offset;
    static const int permute = c2r_power_of_two_constants<2>::permute;
};

template<>
struct c2r_offset_constants<4> {
    static const int offset = c2r_power_of_two_constants<4>::offset;
    static const int permute = c2r_power_of_two_constants<4>::permute;
};

template<>
struct c2r_offset_constants<8> {
    static const int offset = c2r_power_of_two_constants<8>::offset;
    static const int permute = c2r_power_of_two_constants<8>::permute;
};

template<int s>
struct r2c_offset_constants{};

template<>
struct r2c_offset_constants<3> {
    static const int permute = c2r_offset_constants<3>::rotate;
};

template<>
struct r2c_offset_constants<5> {
    static const int permute = c2r_offset_constants<5>::rotate;
};

template<>
struct r2c_offset_constants<7> {
    static const int permute = c2r_offset_constants<7>::rotate;
};

template<>
struct r2c_offset_constants<9> {
    static const int permute = c2r_offset_constants<9>::rotate;
};

template<typename T, typename constants, int size, int position=0>
struct tx_permute_impl{};

template<typename HT, typename TT, typename constants, int size, int position>
struct tx_permute_impl<
    thrust::detail::cons<HT, TT>, constants, size, position> {
    typedef typename homogeneous_tuple<size, HT>::type Source;
    typedef thrust::detail::cons<HT, TT> Remaining;
    static const int permute = constants::permute;
    static const int new_position = (position + permute) % size;
    __host__ __device__
    static Remaining impl(const Source& src) {
        return Remaining(
            thrust::get<position>(src),
            tx_permute_impl<TT, constants, size, new_position>::impl(
                src));
    }
};

template<typename constants, int size, int position>
struct tx_permute_impl<thrust::null_type, constants, size, position> {
    template<typename Source>
    __host__ __device__
    static thrust::null_type impl(const Source&) {
        return thrust::null_type();
    }
};


template<typename Tuple>
__host__ __device__ Tuple c2r_tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        c2r_offset_constants<thrust::tuple_size<Tuple>::value>,
        thrust::tuple_size<Tuple>::value>::impl(t);
}

template<typename Tuple>
__host__ __device__ Tuple r2c_tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        r2c_offset_constants<thrust::tuple_size<Tuple>::value>,
        thrust::tuple_size<Tuple>::value>::impl(t);
}


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

template<int m, bool power_of_two>
struct c2r_compute_initial_offset {
    typedef c2r_offset_constants<m> constants;
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((WARP_SIZE - warp_id) * constants::offset)
            & WARP_MASK;
        return initial_offset;
    }
};

template<int m>
struct c2r_compute_initial_offset<m, true> {
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((warp_id * (WARP_SIZE + 1)) >>
                              static_log<m>::value)
            & WARP_MASK;
        return initial_offset;
    }
};




template<int m, bool power_of_two>
__device__
typename homogeneous_tuple<m, int>::type c2r_compute_offsets() {
    typedef c2r_offset_constants<m> constants;
    typedef typename homogeneous_tuple<m, int>::type result_type;
    int initial_offset = c2r_compute_initial_offset<m, power_of_two>::impl();
    return compute_offsets_impl<result_type,
                                WARP_SIZE,
                                constants::offset>::impl(initial_offset);
}

template<typename Data, typename Indices>
struct warp_shuffle {};

template<typename DHT, typename DTT, typename IHT, typename ITT>
struct warp_shuffle<
    thrust::detail::cons<DHT, DTT>,
    thrust::detail::cons<IHT, ITT> > {
    __device__ static void impl(thrust::detail::cons<DHT, DTT>& d,
                                const thrust::detail::cons<IHT, ITT>& i) {
        d.get_head() = __shfl(d.get_head(), i.get_head());
        warp_shuffle<DTT, ITT>::impl(d.get_tail(),
                                     i.get_tail());
    }
};

template<>
struct warp_shuffle<
    thrust::null_type, thrust::null_type> {
    __device__ static void impl(thrust::null_type, thrust::null_type) {}
};


template<typename IntTuple, bool is_power_of_two>
struct c2r_compute_indices_impl {
    __device__ static void impl(IntTuple& indices, int& rotation) {
        indices =
            detail::c2r_compute_offsets<thrust::tuple_size<IntTuple>::value,
                                        false>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    int r =
        detail::c2r_offset_constants
        <thrust::tuple_size<IntTuple>::value>::rotate;
    rotation = (warp_id * r) % size;
}
};

template<typename IntTuple>
struct c2r_compute_indices_impl<IntTuple, true> {
    __device__ static void impl(IntTuple& indices, int& rotation) {
        indices =
            detail::c2r_compute_offsets<thrust::tuple_size<IntTuple>::value,
                                        true>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    rotation = (size - warp_id) & (size - 1);
}
};

template<typename Tuple, typename IntTuple, bool is_power_of_two>
struct c2r_warp_transpose_impl {
    __device__ static void impl(Tuple& src,
                                const IntTuple& indices,
                                const int& rotation) {
        detail::warp_shuffle<Tuple, IntTuple>::impl(src, indices);
        src = rotate(detail::c2r_tx_permute(src), rotation);
    }
};

template<typename Tuple, typename IntTuple>
struct c2r_warp_transpose_impl<Tuple, IntTuple, true> {
    __device__ static void impl(Tuple& src,
                                const IntTuple& indices,
                                const int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
        int pre_rotation = warp_id >>
            (LOG_WARP_SIZE -
             static_log<thrust::tuple_size<Tuple>::value>::value);
        src = rotate(src, pre_rotation);        
        c2r_warp_transpose_impl<Tuple, IntTuple, false>::impl
            (src, indices, rotation);
    }
};

} //end namespace detail

template<typename IntTuple>
__device__ void c2r_compute_indices(IntTuple& indices, int& rotation) {
    detail::c2r_compute_indices_impl<
        IntTuple,
        is_power_of_two<thrust::tuple_size<IntTuple>::value>::value>
        ::impl(indices, rotation);
}

template<typename Tuple>
__device__ void c2r_warp_transpose(Tuple& src,
                                   const typename homogeneous_tuple<
                                       thrust::tuple_size<Tuple>::value,
                                       int>::type& indices,
                                   int rotation) {    
    typedef typename
        homogeneous_tuple<thrust::tuple_size<Tuple>::value, int>::type
        IntTuple;
    detail::c2r_warp_transpose_impl<
        Tuple, IntTuple,
        is_power_of_two<thrust::tuple_size<Tuple>::value>::value>::
        impl(src, indices, rotation);
}

}
