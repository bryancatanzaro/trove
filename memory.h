#pragma once
#include "utility.h"

namespace trove {
namespace detail {

template<typename Tuple>
struct warp_store_tuple {};

template<typename HT, typename TT>
struct warp_store_tuple<thrust::detail::cons<HT, TT> > {
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        HT* ptr, int offset, int stride) {
        ptr[offset] = d.get_head();
        warp_store_tuple<TT>::impl(d.get_tail(), ptr, offset + stride, stride);
    }
};

template<>
struct warp_store_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, T*, int, int) {}
};

template<typename T>
struct uncoalesced_store_tuple{};

template<typename HT, typename TT>
struct uncoalesced_store_tuple<thrust::detail::cons<HT, TT> > {
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        HT* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.get_head();
        uncoalesced_store_tuple<TT>::impl(d.get_tail(), ptr, offset+1,
            stride);
    }
    __host__ __device__ static void impl(
        const thrust::detail::cons<HT, TT>& d,
        volatile HT* ptr,
        int offset=0, int stride=1) {
        ptr[offset] = d.get_head();
        uncoalesced_store_tuple<TT>::impl(d.get_tail(), ptr, offset+1,
            stride);
    }
};

template<>
struct uncoalesced_store_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, T*, int, int) {}
    template<typename T>
    __host__ __device__ static void impl(
        thrust::null_type, volatile T*, int, int) {}
};

template<typename T>
struct warp_load_tuple{};

template<typename HT, typename TT>
struct warp_load_tuple<thrust::detail::cons<HT, TT> > {
    typedef thrust::detail::cons<HT, TT> Tuple;
    __host__ __device__ static Tuple impl(HT* ptr,
                                          int offset,
                                          int stride=32) {
        return Tuple(ptr[offset],
                     warp_load_tuple<TT>::impl(ptr, offset+stride, stride));
    }
    __host__ __device__ static Tuple impl(volatile HT* ptr,
                                          int offset,
                                          int stride=32) {
        return Tuple(ptr[offset],
                     warp_load_tuple<TT>::impl(ptr, offset+stride, stride));
    }
};

template<>
struct warp_load_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__ static thrust::null_type impl(
        T*, int, int) {
        return thrust::null_type();
    }
    template<typename T>
    __host__ __device__ static thrust::null_type impl(
        volatile T*, int, int) {
        return thrust::null_type();
    }
};

} //end namespace detail

template<typename Tuple>
__host__ __device__ void warp_store(const Tuple& t,
                                    typename Tuple::head_type* ptr,
                                    int offset, int stride=32) {
    detail::warp_store_tuple<
        typename cons_type<Tuple>::type>::impl(t, ptr, offset, stride);
}

template<typename Tuple>
__host__ __device__ Tuple warp_load(typename Tuple::head_type* ptr,
                                    int offset, int stride=32) {
    return detail::warp_load_tuple<
        typename cons_type<Tuple>::type>::impl(ptr, offset, stride);
}

template<typename Tuple>
__host__ __device__ Tuple warp_load(volatile typename Tuple::head_type* ptr,
                                    int offset, int stride=32) {
    return detail::warp_load_tuple<
        typename cons_type<Tuple>::type>::impl(ptr, offset, stride);
}


template<typename Tuple>
__host__ __device__ void uncoalesced_store(const Tuple& t,
                                           typename Tuple::head_type* ptr,
                                           int stride=1) {
    detail::uncoalesced_store_tuple<
        typename cons_type<Tuple>::type>::impl(t, ptr, 0, stride);
}

template<typename Tuple>
__host__ __device__ void uncoalesced_store(
    const Tuple& t,
    volatile typename Tuple::head_type* ptr,
    int stride=1) {
    detail::uncoalesced_store_tuple<
        typename cons_type<Tuple>::type>::impl(t, ptr, 0, stride);
}

} //end namespace trove
