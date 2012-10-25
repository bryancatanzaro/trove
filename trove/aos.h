#pragma once
#include <trove/memory.h>
#include <trove/detail/dismember.h>
#include <trove/transpose.h>
#include <trove/utility.h>

namespace trove {

namespace detail {

template<typename T, typename I>
__device__ T load_aos_warp_contiguous(const T* src, const I& idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    I warp_begin_idx = idx - warp_id;
    const T* warp_begin_src = src + warp_begin_idx;
    const int* as_int_src = (const int*)warp_begin_src;
    typedef array<int, detail::size_in_ints<T>::value> int_store;
    int_store loaded = warp_load<int_store>(as_int_src, warp_id);
    r2c_warp_transpose(loaded);
    return detail::fuse<T>(loaded);
}

template<typename T, typename I>
__device__ void store_aos_warp_contiguous(const T& data, T* dest, const I& idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    I warp_begin_idx = idx - warp_id;
    T* warp_begin_dest = dest + warp_begin_idx;
    int* as_int_dest = (int*)warp_begin_dest;
    typedef array<int, detail::size_in_ints<T>::value> int_store;
    int_store lysed = detail::lyse(data);
    c2r_warp_transpose(lysed);
    warp_store(lysed, as_int_dest, warp_id);
    
}

template<typename T, typename I>
__device__ int* compute_address(T* src, const I& src_index, int impl_index) {
    int shuffle_index = impl_index / size_in_ints<T>::value;
    int sub_index = impl_index % size_in_ints<T>::value;
    I base_index = __shfl(src_index, shuffle_index);
    int* result = (base_index < 0) ? NULL :
                   ((int*)(src + base_index) + sub_index);
    return result;
}
        
template<int s, typename T, typename I>
struct indexed_load {
    __device__
    static array<int, s> impl(const T* src,
                              const I& src_index, int impl_index) {
        int result;
        int* address = compute_address(src, src_index, impl_index);
        if (address != NULL) result = *address;
        return array<int, s>(
            result,
            indexed_load<s-1, T, I>::impl(src, src_index,
                                          impl_index + WARP_SIZE));
    }
};

template<typename T, typename I>
struct indexed_load<1, T, I> {
    __device__
    static array<int, 1> impl(const T* src,
                              const I& src_index, int impl_index) {
        int result;
        int* address = compute_address(src, src_index, impl_index);
        if (address != NULL) result = *address;
        return array<int, 1>(result);
    }
};

template<int s, typename T, typename I>
struct indexed_store {
    __device__
    static void impl(const array<int, s>& src,
                     T* dest, const I& dest_index, int impl_index) {
        int* address = compute_address(dest, dest_index, impl_index);
        if (address != NULL) *address = src.head;
        indexed_store<s-1, T, I>::impl(src.tail, dest, dest_index,
                                       impl_index + WARP_SIZE);
    }
};

template<typename T, typename I>
struct indexed_store<1, T, I> {
    __device__
    static void impl(const array<int, 1>& src,
                     T* dest, const I& dest_index, int impl_index) {
        int* address = compute_address(dest, dest_index, impl_index);
        if (address != NULL) *address = src.head;
    }
};

template<typename I>
__device__
bool is_contiguous(int warp_id, const I& idx) {
    int neighbor_idx = (warp_id == 0) ? 0 : warp_id-1;
    I neighbor = __shfl(idx, neighbor_idx);
    bool neighbor_contiguous = (warp_id == 0) ? idx > 0 : (idx - neighbor == 1);
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

template<typename T, typename I>
__device__ typename enable_if<use_shfl<T>::value, T>::type
load_aos_dispatch(const T* src, const I& idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    if (detail::is_contiguous(warp_id, idx)) {
        return detail::load_aos_warp_contiguous(src, idx);
    } else {
        typedef array<int, detail::size_in_ints<T>::value> int_store;
        int_store loaded =
            detail::indexed_load<detail::size_in_ints<T>::value,
                                 T, I>::impl(src, idx, warp_id);
        r2c_warp_transpose(loaded);
        if (idx >= 0) return detail::fuse<T>(loaded);
        else return T();
    }   
}

template<typename T, typename I>
__device__ typename enable_if<!use_shfl<T>::value, T>::type
load_aos_dispatch(const T* src, const I& idx) {
    if (idx >= 0)
        return src[idx];
    else
        return T();
}

template<typename T, typename I>
__device__ typename enable_if<use_shfl<T>::value>::type
store_aos_dispatch(const T& data, T* dest, const I& idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    if (detail::is_contiguous(warp_id, idx)) {
        detail::store_aos_warp_contiguous(data, dest, idx);
    } else {
        typedef array<int, detail::size_in_ints<T>::value> int_store;
        int_store lysed = detail::lyse(data);
        c2r_warp_transpose(lysed);
        detail::indexed_store<detail::size_in_ints<T>::value,
                              T, I>::impl(lysed, dest, idx, warp_id);
    }
}

template<typename T, typename I>
__device__ typename enable_if<!use_shfl<T>::value>::type
store_aos_dispatch(const T& data, T* dest, const I& idx) {
    if (idx >= 0) dest[idx] = data;
}


}

template<typename T, typename I>
__device__ T load_aos(const T* src, const I& idx) {
    return detail::load_aos_dispatch(src, idx);
}

template<typename T, typename I>
__device__ void store_aos(const T& data, T* dest, const I& idx) {
    detail::store_aos_dispatch(data, dest, idx);
}

}
