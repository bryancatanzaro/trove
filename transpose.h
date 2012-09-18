#pragma once
#include "utility.h"
#include "rotate.h"

namespace trove {
namespace detail {

template<int s>
struct offset_constants{};

// This Python code computes the necessary magic constants for arbitrary sizes
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
struct offset_constants<3> {
    static const int offset=21;
    static const int permute=2;
    static const int rotate=2;
};

template<>
struct offset_constants<5> {
    static const int offset=19;
    static const int permute=2;
    static const int rotate=3;
};

template<>
struct offset_constants<7> {
    static const int offset=9;
    static const int permute=4;
    static const int rotate=2;
};

template<>
struct offset_constants<9> {
    static const int offset=7;
    static const int permute=5;
    static const int rotate=2;
};


template<typename T, int size, int position=0>
struct tx_permute_impl{};

template<typename HT, typename TT, int size, int position>
struct tx_permute_impl<thrust::detail::cons<HT, TT>, size, position> {
    typedef typename homogeneous_tuple<size, HT>::type Source;
    typedef thrust::detail::cons<HT, TT> Remaining;
    static const int permute = offset_constants<size>::permute;
    static const int new_position = (position + permute) % size;
    __host__ __device__
    static Remaining impl(const Source& src) {
        return Remaining(thrust::get<position>(src),
                         tx_permute_impl<TT, size, new_position>::impl(
                             src));
    }
};

template<int size, int position>
struct tx_permute_impl<thrust::null_type, size, position> {
    template<typename Source>
    __host__ __device__
    static thrust::null_type impl(const Source&) {
        return thrust::null_type();
    }
};


template<typename Tuple>
__host__ __device__ Tuple tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        thrust::tuple_size<Tuple>::value>::impl(t);
}



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

} //end namespace detail

template<typename IntTuple>
__device__ void compute_indices(IntTuple& indices, int& rotation) {
    indices =
        detail::compute_offsets<thrust::tuple_size<IntTuple>::value>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    int r =
        detail::offset_constants<thrust::tuple_size<IntTuple>::value>::rotate;
    rotation = (warp_id * r) % size;
}

template<typename Tuple>
__device__ void warp_transpose(Tuple& src,
                               const typename homogeneous_tuple<
                                   thrust::tuple_size<Tuple>::value,
                                   int>::type& indices,
                               int rotation) {
    typedef typename
        homogeneous_tuple<thrust::tuple_size<Tuple>::value, int>::type
        IntTuple;
    detail::warp_shuffle<Tuple, IntTuple>::impl(src, indices);
    src = rotate(detail::tx_permute(src), rotation);
}

}
