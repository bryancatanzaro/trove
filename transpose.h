#pragma once
#include "utility.h"
#include "rotate.h"
#include "shfl.h"
#include "static_mod_inverse.h"
#include "static_gcd.h"

#define WARP_SIZE 32
#define WARP_MASK 0x1f
#define LOG_WARP_SIZE 5

namespace trove {
namespace detail {

struct odd{};
struct power_of_two{};
struct composite{};

template<int m, bool ispo2=is_power_of_two<m>::value, bool isodd=is_odd<m>::value>
struct tx_algorithm {
    typedef composite type;
};

template<int m>
struct tx_algorithm<m, true, false> {
    typedef power_of_two type;
};

template<int m>
struct tx_algorithm<m, false, true> {
    typedef odd type;
};

template<int m, typename Schema=typename tx_algorithm<m>::type>
struct c2r_offset_constants{};

template<int m>
struct c2r_offset_constants<m, odd> {
    static const int offset = WARP_SIZE - static_mod_inverse<m, WARP_SIZE>::value;
    static const int rotate = static_mod_inverse<WARP_SIZE, m>::value;
    static const int permute = static_mod_inverse<rotate, m>::value;
};

template<int m>
struct c2r_offset_constants<m, power_of_two> {
    static const int offset = WARP_SIZE - WARP_SIZE/m; 
    static const int permute = m - 1;
};

template<int m>
struct c2r_offset_constants<m, composite> {
    static const int c = static_gcd<m, WARP_SIZE>::value;
    static const int k = static_mod_inverse<m/c, WARP_SIZE/c>::value;
    static const int permute = WARP_SIZE % m;
    static const int period = m / c;
};

template<int m, typename Schema=typename tx_algorithm<m>::type>
struct r2c_offset_constants{};

template<int m>
struct r2c_offset_constants<m, odd> {
    static const int permute = static_mod_inverse<WARP_SIZE, m>::value;
};

template<int m>
struct r2c_offset_constants<m, power_of_two> :
    c2r_offset_constants<m, power_of_two> {
};

template<typename T, template<int> class Permute, int position=0>
struct tx_permute_impl{};

template<typename HT, typename TT, template<int> class Permute, int position>
struct tx_permute_impl<
    thrust::detail::cons<HT, TT>, Permute, position> {
    typedef thrust::detail::cons<HT, TT> Remaining;
    static const int idx = Permute<position>::value;
    template<typename Source>
    __host__ __device__
    static Remaining impl(const Source& src) {
        return Remaining(
            thrust::get<idx>(src),
            tx_permute_impl<TT, Permute, position+1>::impl(
                src));
    }
};

template<template<int> class Permute, int position>
struct tx_permute_impl<thrust::null_type, Permute, position> {
    template<typename Source>
    __host__ __device__
    static thrust::null_type impl(const Source&) {
        return thrust::null_type();
    }
};


template<int m, int a, int b=0>
struct affine_modular_fn {
    template<int x>
    struct eval {
        static const int value = (a * x + b) % m;
    };
};

template<int m>
struct composite_c2r_permute_fn {
    static const int o = WARP_SIZE % m;
    static const int c = static_gcd<m, WARP_SIZE>::value;
    static const int p = m / c;
    template<int x>
    struct eval {
        static const int value = (x * o - (x / p)) % m;
    };
};




template<typename Tuple>
__host__ __device__ Tuple c2r_tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        affine_modular_fn<thrust::tuple_size<Tuple>::value,
                          c2r_offset_constants<thrust::tuple_size<Tuple>::value>::permute>::template eval>::impl(t);
}



template<typename Tuple>
__host__ __device__ Tuple composite_c2r_tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        composite_c2r_permute_fn<thrust::tuple_size<Tuple>::value>::template eval>::impl(t);
}


template<typename Tuple>
__host__ __device__ Tuple r2c_tx_permute(const Tuple& t) {
    return tx_permute_impl<
        typename cons_type<Tuple>::type,
        affine_modular_fn<thrust::tuple_size<Tuple>::value,
                          r2c_offset_constants<thrust::tuple_size<Tuple>::value>::permute>::template eval>::impl(t);
}


template<typename IntTuple, int b, int o>
struct c2r_compute_offsets_impl{};

template<typename HT, typename TT, int b, int o>
struct c2r_compute_offsets_impl<thrust::detail::cons<HT, TT>, b, o> {
    typedef thrust::detail::cons<HT, TT> Tuple;
    __device__
    static Tuple impl(int offset) {
        if (offset >= b) {
            offset -= b;
        } //Poor man's x % b. Requires that o < b.
        return Tuple(offset,
                     c2r_compute_offsets_impl<TT, b, o>::
                     impl(offset + o));
    }
};

template<int b, int o>
struct c2r_compute_offsets_impl<thrust::null_type, b, o> {
    __device__
    static thrust::null_type impl(int) {
        return thrust::null_type();
    }
};

template<int m, typename Schema>
struct c2r_compute_initial_offset {};

template<int m>
struct c2r_compute_initial_offset<m, odd> {
    typedef c2r_offset_constants<m> constants;
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((WARP_SIZE - warp_id) * constants::offset)
            & WARP_MASK;
        return initial_offset;
    }
};

template<int m>
struct c2r_compute_initial_offset<m, power_of_two> {
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((warp_id * (WARP_SIZE + 1)) >>
                              static_log<m>::value)
            & WARP_MASK;
        return initial_offset;
    }
};

template<int m, typename Schema>
struct r2c_compute_initial_offset {};

template<int m>
struct r2c_compute_initial_offset<m, odd> {
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = (warp_id * m) & WARP_MASK;
        return initial_offset;
    }
};


template<int m, typename Schema>
__device__
typename homogeneous_tuple<m, int>::type c2r_compute_offsets() {
    typedef c2r_offset_constants<m> constants;
    typedef typename homogeneous_tuple<m, int>::type result_type;
    int initial_offset = c2r_compute_initial_offset<m, Schema>::impl();
    return c2r_compute_offsets_impl<result_type,
                                    WARP_SIZE,
                                    constants::offset>::impl(initial_offset);
}

template<int index, int offset, int bound>
struct r2c_offsets {
    static const int value = (offset * index) % bound;
};

template<typename IntTuple, int index, int m, typename Schema>
struct r2c_compute_offsets_impl{};

template<typename HT, typename TT, int index, int m, typename Schema>
struct r2c_compute_offsets_impl<thrust::detail::cons<HT, TT>, index, m, Schema> {
    typedef thrust::detail::cons<HT, TT> Tuple;
    static const int offset = (WARP_SIZE % m * index) % m;
    __device__
    static Tuple impl(int initial_offset) {
        int current_offset = (initial_offset + offset) & WARP_MASK;
        return Tuple(current_offset,
                     r2c_compute_offsets_impl<TT, index + 1, m, Schema>::
                     impl(initial_offset));
    }
};

template<typename HT, typename TT, int index, int m>
struct r2c_compute_offsets_impl<thrust::detail::cons<HT, TT>, index, m, power_of_two> {
  typedef thrust::detail::cons<HT, TT> Tuple;
  __device__
  static Tuple impl(int initial_offset) {
    int warp_id = threadIdx.x & WARP_MASK;
    
    const int logL = static_log<m>::value;
    const int logP = LOG_WARP_SIZE;
    int msb_bits = warp_id >> (logP - logL);
    int lsb_bits = warp_id & ((1 << (logP - logL)) - 1);

    return Tuple((lsb_bits << logL) | ((msb_bits + m - index) % m),
                 r2c_compute_offsets_impl<TT, index + 1, m, power_of_two>::impl(initial_offset));
  }
};

template<int index, int m, typename Schema>
struct r2c_compute_offsets_impl<thrust::null_type, index, m, Schema> {
    __device__
    static thrust::null_type impl(int) {
        return thrust::null_type();
    }
};


template<int m, typename Schema>
__device__
typename homogeneous_tuple<m, int>::type r2c_compute_offsets() {
    typedef r2c_offset_constants<m> constants;
    typedef typename homogeneous_tuple<m, int>::type result_type;
    int initial_offset = r2c_compute_initial_offset<m, Schema>::impl();
    return r2c_compute_offsets_impl<result_type,
                                    0, m, Schema>::impl(initial_offset);
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


template<typename IntTuple, typename Schema>
struct c2r_compute_indices_impl {};

template<typename IntTuple>
struct c2r_compute_indices_impl<IntTuple, odd> {
    __device__ static void impl(IntTuple& indices, int& rotation) {
        indices =
            detail::c2r_compute_offsets<thrust::tuple_size<IntTuple>::value,
                                        odd>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    int r =
        detail::c2r_offset_constants
        <thrust::tuple_size<IntTuple>::value>::rotate;
    rotation = (warp_id * r) % size;
    }
};

template<typename IntTuple>
struct c2r_compute_indices_impl<IntTuple, power_of_two> {
    __device__ static void impl(IntTuple& indices, int& rotation) {
        indices =
            detail::c2r_compute_offsets<thrust::tuple_size<IntTuple>::value,
                                        power_of_two>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    rotation = (size - warp_id) & (size - 1);
    }
};

template<typename Tuple, typename IntTuple, typename Schema>
struct c2r_warp_transpose_impl {};

template<typename Tuple, typename IntTuple>
struct c2r_warp_transpose_impl<Tuple, IntTuple, odd> {
    __device__ static void impl(Tuple& src,
                                const IntTuple& indices,
                                const int& rotation) {
        detail::warp_shuffle<Tuple, IntTuple>::impl(src, indices);
        src = rotate(detail::c2r_tx_permute(src), rotation);
    }
};

template<typename Tuple, typename IntTuple>
struct c2r_warp_transpose_impl<Tuple, IntTuple, power_of_two> {
    __device__ static void impl(Tuple& src,
                                const IntTuple& indices,
                                const int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
        int pre_rotation = warp_id >>
            (LOG_WARP_SIZE -
             static_log<thrust::tuple_size<Tuple>::value>::value);
        src = rotate(src, pre_rotation);        
        c2r_warp_transpose_impl<Tuple, IntTuple, odd>::impl
            (src, indices, rotation);
    }
};

template<typename IntTuple, typename Schema>
struct r2c_compute_indices_impl {};

template<typename IntTuple>
struct r2c_compute_indices_impl<IntTuple, odd> {
    __device__ static void impl(IntTuple& indices, int& rotation) {
        indices =
            detail::r2c_compute_offsets<thrust::tuple_size<IntTuple>::value,
                                        odd>();
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    int r =
        size - detail::r2c_offset_constants
        <thrust::tuple_size<IntTuple>::value>::permute;
    rotation = (warp_id * r) % size;
    }
};

template<typename IntTuple>
struct r2c_compute_indices_impl<IntTuple, power_of_two> {
  __device__ static void impl(IntTuple& indices, int& rotation) {
    int warp_id = threadIdx.x & WARP_MASK;
    int size = thrust::tuple_size<IntTuple>::value;
    rotation = warp_id % size;
    indices = r2c_compute_offsets_impl<IntTuple, 0,
                                       thrust::tuple_size<IntTuple>::value,
                                       power_of_two>::impl(0);
  }
};

template<typename Tuple, typename IntTuple, typename Schema>
struct r2c_warp_transpose_impl {};

template<typename Tuple, typename IntTuple>
struct r2c_warp_transpose_impl<Tuple, IntTuple, odd> {
    __device__ static void impl(Tuple& src,
                                const IntTuple& indices,
                                const int& rotation) {
        Tuple rotated = rotate(src, rotation);
        detail::warp_shuffle<Tuple, IntTuple>::impl(rotated, indices);
        src = detail::r2c_tx_permute(rotated);
    }
};

template<typename Tuple, typename IntTuple>
struct r2c_warp_transpose_impl<Tuple, IntTuple, power_of_two> {
  __device__ static void impl(Tuple& src,
                              const IntTuple& indices,
                              const int& rotation) {
    Tuple rotated = rotate(src, rotation);
    detail::warp_shuffle<Tuple, IntTuple>::impl(rotated, indices);
    const int size = thrust::tuple_size<IntTuple>::value;
    int warp_id = threadIdx.x & WARP_MASK;
    src = rotate(detail::r2c_tx_permute(rotated), (size-warp_id/(WARP_SIZE/size))%size);
  }
};

} //end namespace detail

template<typename IntTuple>
__device__ void c2r_compute_indices(IntTuple& indices, int& rotation) {
    detail::c2r_compute_indices_impl<
        IntTuple,
        typename detail::tx_algorithm<thrust::tuple_size<IntTuple>::value>::type>
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
        typename detail::tx_algorithm<thrust::tuple_size<Tuple>::value>::type>::
        impl(src, indices, rotation);
}

template<typename IntTuple>
__device__ void r2c_compute_indices(IntTuple& indices, int& rotation) {
    detail::r2c_compute_indices_impl<
        IntTuple,
        typename detail::tx_algorithm<thrust::tuple_size<IntTuple>::value>::type>::impl(indices, rotation);

}

template<typename Tuple>
__device__ void r2c_warp_transpose(Tuple& src,
                                   const typename homogeneous_tuple<
                                   thrust::tuple_size<Tuple>::value,
                                   int>::type& indices,
                                   int rotation) {
    typedef typename
        homogeneous_tuple<thrust::tuple_size<Tuple>::value, int>::type
        IntTuple;
    detail::r2c_warp_transpose_impl<
        Tuple, IntTuple,
        typename detail::tx_algorithm<thrust::tuple_size<Tuple>::value>::type>::
        impl(src, indices, rotation);
}

} //end namespace trove
