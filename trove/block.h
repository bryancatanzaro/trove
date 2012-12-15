#pragma once
#include <trove/aos.h>

namespace trove {


template<int s, typename T, typename I>
__device__
trove::array<T, s> load_array_warp_contiguous(const T* src, const I& idx) {
    typedef trove::array<T, s> array_type;
    const array_type* src_ptr = (const array_type*)(src) + idx;
    return load_warp_contiguous(src_ptr);
}

template<int s, typename T, typename I>
__device__
trove::array<T, s> load_array(const T* src, const I& idx) {
    typedef trove::array<T, s> array_type;
    const array_type* src_ptr = (const array_type*)(src) + idx;
    return *src_ptr;
}

template<int s, typename T, typename I>
__device__
void store_array_warp_contiguous(T* dest, const I& idx, const trove::array<T, s>& src) {
    typedef trove::array<T, s> array_type;
    array_type* dest_ptr = (array_type*)(dest) + idx;
    store_warp_contiguous(src, dest_ptr);
}

template<int s, typename T, typename I>
__device__
void store_array(T* dest, const I& idx, const trove::array<T, s>& src) {
    typedef trove::array<T, s> array_type;
    array_type* dest_ptr = (array_type*)(dest) + idx;
    *dest_ptr = src;
}

}
