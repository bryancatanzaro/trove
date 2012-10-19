#pragma once
#include <trove/array.h>
#include <trove/detail/dismember.h>

__device__
int __shfl(const int& t, const int& i, int* smem) {
    const int warp_begin = threadIdx.x & (~0x1f);
    smem[threadIdx.x] = t;
    return smem[warp_begin + i];
}

namespace trove {
namespace detail {

template<int s>
struct shuffle {
    __device__
    static void impl(array<int, s>& d, const int& i, int* smem) {
        d.head = __shfl(d.head, i, smem);
        shuffle<s-1>::impl(d.tail, i, smem);
    }
};

template<>
struct shuffle<1> {
    __device__
    static void impl(array<int, 1>& d, const int& i, int* smem) {
        d.head = __shfl(d.head, i, smem);
    }
};
 
}
}

template<typename T>
__device__
T __shfl(const T& t, const int& i, int* smem) {
    typedef trove::array<int, trove::detail::size_in_ints<T>::value>
        lysed_array;
    lysed_array lysed = trove::detail::lyse(t);
    trove::detail::shuffle<lysed_array>
        ::impl(lysed, i, smem);
    return trove::detail::fuse<T>(lysed);
}
