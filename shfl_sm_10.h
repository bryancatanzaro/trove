#pragma once

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
    static void impl(int d[s], const int& i, int* smem) {
        d[0] = __shfl(d[0], i, smem);
        shuffle<s-1>::impl(d+1, i, smem);
    }
};

template<>
struct shuffle<1> {
    __device__
        static void impl(int d[1], const int& i, int* smem) {
        d[0] = __shfl(d[0], i, smem);
    }
};
 
}
}

template<typename T>
__device__
T __shfl(const T& t, const int& i, int* smem) {
    union trove::detail::dismember<T,
    trove::detail::size_in_ints<T>::value> cleaver;
    cleaver.d = t;
    trove::detail::shuffle<trove::detail::size_in_ints<T>::value>
      ::impl(cleaver.i, i, smem);
    return cleaver.d;
}
