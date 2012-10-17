#pragma once
#include "memory.h"
#include "dismember.h"
#include "transpose.h"

namespace trove {

template<typename T, typename I>
__device__ T load_warp_contiguous(T* src, I idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    I warp_begin_idx = idx - warp_id;
    T* warp_begin_src = src + warp_begin_idx;
    int* as_int_src = (int*)warp_begin_src;
    typedef array<int, detail::size_in_ints<T>::value> int_store;
    int_store loaded = warp_load<int_store>(as_int_src, warp_id);
    r2c_warp_transpose(loaded);
    return fuse<T>(loaded);
}

template<typename T, typename I>
__device__ void store_warp_contiguous(const T& data, T* dest, I idx) {
    int warp_id = threadIdx.x & WARP_MASK;
    I warp_begin_idx = idx - warp_id;
    T* warp_begin_dest = dest + warp_begin_idx;
    int* as_int_dest = (int*)warp_begin_dest;
    typedef array<int, detail::size_in_ints<T>::value> int_store;
    int_store lysed = lyse(data);
    c2r_warp_transpose(lysed);
    warp_store(lysed, as_int_dest, warp_id);
    
}

}
