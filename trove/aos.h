#pragma once
#include <trove/detail/dismember.h>
#include <trove/transpose.h>
#include <trove/utility.h>

//#define WARP_CONVERGED ((1LL << WARP_SIZE) - 1)
#define WARP_CONVERGED 0xffffffff

namespace trove {

namespace detail {

template<typename T>
__device__ int* compute_address(T* src, int impl_index) {
    int shuffle_index = impl_index / size_in_ints<T>::value;
    int sub_index = impl_index % size_in_ints<T>::value;
    T* base_ptr = __shfl(src, shuffle_index);
    int* result = ((int*)(base_ptr) + sub_index);
    return result;
}
        
template<int s, typename T>
struct indexed_load {
    __device__
    static array<int, s> impl(const T* src, int impl_index) {
        int result;
        int* address = compute_address(src, impl_index);
        result = *address;
        return array<int, s>(
            result,
            indexed_load<s-1, T>::impl(src, impl_index + WARP_SIZE));
    }
};

template<typename T>
struct indexed_load<1, T> {
    __device__
    static array<int, 1> impl(const T* src, int impl_index) {
        int result;
        int* address = compute_address(src, impl_index);
        result = *address;
        return array<int, 1>(result);
    }
};

template<int s, typename T>
struct indexed_store {
    __device__
    static void impl(const array<int, s>& src,
                     T* dest, int impl_index) {
        int* address = compute_address(dest, impl_index);
        *address = src.head;
        indexed_store<s-1, T>::impl(src.tail, dest, impl_index + WARP_SIZE);
    }
};

template<typename T>
struct indexed_store<1, T> {
    __device__
    static void impl(const array<int, 1>& src,
                     T* dest, int impl_index) {
        int* address = compute_address(dest, impl_index);
        *address = src.head;
    }
};

template<typename T>
__device__
bool is_contiguous(int warp_id, const T* ptr) {
    int neighbor_idx = (warp_id == 0) ? 0 : warp_id-1;
    const T* neighbor_ptr = __shfl(ptr, neighbor_idx);
    bool neighbor_contiguous = (warp_id == 0) ? true : (ptr - neighbor_ptr == sizeof(T));
    bool result = __all(neighbor_contiguous);
    return result;
}


template<typename T>
struct size_multiple_four {
    static const bool value = (sizeof(T) & 0x3) == 0;
};

template<typename T>
struct size_in_range {
    static const bool value = (sizeof(T) >= 8) && (sizeof(T) <= 252);
};

template<typename T, bool s=size_multiple_four<T>::value, bool r=size_in_range<T>::value>
struct use_shfl {
    static const bool value = false;
};

template<typename T>
struct use_shfl<T, true, true> {
    static const bool value = true;
};

template<typename T>
__device__ typename enable_if<use_shfl<T>::value, T>::type
load_dispatch(const T* src) {
    int warp_id = threadIdx.x & WARP_MASK;
    // if (detail::is_contiguous(warp_id, src)) {
    //     return detail::load_warp_contiguous(src);
    // } else {
        typedef array<int, detail::size_in_ints<T>::value> int_store;
        int_store loaded =
            detail::indexed_load<detail::size_in_ints<T>::value, T>::impl(
                src, warp_id);
        r2c_warp_transpose(loaded);
        return detail::fuse<T>(loaded);
    // }   
}

template<typename T>
__device__ typename enable_if<!use_shfl<T>::value, T>::type
load_dispatch(const T* src) {
    return *src;
}

template<typename T>
__device__ typename enable_if<use_shfl<T>::value>::type
store_dispatch(const T& data, T* dest) {
    int warp_id = threadIdx.x & WARP_MASK;
    // if (detail::is_contiguous(warp_id, dest)) {
    //     detail::store_warp_contiguous(data, dest);
    // } else {
        typedef array<int, detail::size_in_ints<T>::value> int_store;
        int_store lysed = detail::lyse(data);
        c2r_warp_transpose(lysed);
        detail::indexed_store<detail::size_in_ints<T>::value, T>::impl(
            lysed, dest, warp_id);
    // }
}

template<typename T>
__device__ typename enable_if<!use_shfl<T>::value>::type
store_dispatch(const T& data, T* dest) {
    *dest = data;
}

__device__
bool is_converged() {
    return (__ballot(true) == WARP_CONVERGED);
}
    
}

template<typename T>
__device__ T load(const T* src) {
    if (detail::is_converged()) {
        return detail::load_dispatch(src);
    } else {
        return *src;
    }
}

template<typename T>
__device__ void store(const T& data, T* dest) {
    if (detail::is_converged()) {
        detail::store_dispatch(data, dest);
    } else {
        *dest = data;
    }
}

}
