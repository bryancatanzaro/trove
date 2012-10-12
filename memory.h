#pragma once
#include "utility.h"

namespace trove {
namespace detail {

template<typename Array>
struct warp_store_array {};

template<typename T, int s>
struct warp_store_array<array<T, s> > {
    __host__ __device__ static void impl(
        const array<T, s>& d,
        HT* ptr, int offset, int stride) {
        ptr[offset] = d.head;
        warp_store_array<array<T, s-1> >::impl(
            d.tail, ptr, offset + stride, stride);
    }
};

template<typename T>
struct warp_store_array<array<T, 0> > {
    __host__ __device__ static void impl(
        array<T, 0>, T*, int, int) {}
};

template<typename Array>
struct uncoalesced_store_array{};

template<typename T, int s>
struct uncoalesced_store_array<array<T, s> > {
    __host__ __device__ static void impl(
        const array<T, s>& d,
        T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
        uncoalesced_store_tuple<array<T, s-1> >::impl(d.tail, ptr, offset+1,
                                                      stride);
    }
    __host__ __device__ static void impl(
        const array<T, s>& d,
        volatile T* ptr,
        int offset=0,
        int stride=1) {
        ptr[offset] = d.head;
        uncoalesced_store_tuple<array<T, s-1> >::impl(d.tail, ptr, offset+1,
                                                      stride);
    }
};

template<typename T>
struct uncoalesced_store_array<array<T, 0> > {
    __host__ __device__ static void impl(
        array<T, 0>, T*, int, int) {}
    __host__ __device__ static void impl(
        array<T, 0>, volatile T*, int, int) {}
};

template<typename Array>
struct warp_load_tuple{};

template<typename T, int s>
struct warp_load_tuple<array<T, s> > {
    __host__ __device__ static array<T, s> impl(T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, s>(ptr[offset],
                           warp_load_tuple<array<T, s-1> >::impl(ptr, offset+stride, stride));
    }
    __host__ __device__ static array<T, s> impl(volatile T* ptr,
                                                int offset,
                                                int stride=32) {
        return array<T, s>(ptr[offset],
                           warp_load_tuple<array<T, s-1> >::impl(ptr, offset+stride, stride));
    }
};

template<typename T>
struct warp_load_tuple<array<T, 0> > {
    __host__ __device__ static array<T, 0> impl(
        T*, int, int) {
        return array<T, 0>();
    }
    __host__ __device__ static array<T, 0> impl(
        volatile T*, int, int) {
        return array<T, 0>();
    }
};

} //end namespace detail

template<typename Array>
__host__ __device__ void warp_store(const Array& t,
                                    typename Array::head_type* ptr,
                                    int offset, int stride=32) {
    detail::warp_store_array<Array>::impl(t, ptr, offset, stride);
}

template<typename Array>
__host__ __device__ Array warp_load(typename Array::head_type* ptr,
                                    int offset, int stride=32) {
    return detail::warp_load_array<Array>::impl(ptr, offset, stride);
}

template<typename Array>
__host__ __device__ Array warp_load(volatile typename Array::head_type* ptr,
                                    int offset, int stride=32) {
    return detail::warp_load_array<Array>::impl(ptr, offset, stride);
}

template<typename Array>
__host__ __device__ void uncoalesced_store(const Array& t,
                                           typename Array::head_type* ptr,
                                           int stride=1) {
    detail::uncoalesced_store_array<Array>::impl(t, ptr, 0, stride);
}

template<typename Array>
__host__ __device__ void uncoalesced_store(const Array& t,
                                           volatile typename Array::head_type* ptr,
                                           int stride=1) {
    detail::uncoalesced_store_array<Array>::impl(t, ptr, 0, stride);
}

} //end namespace trove
