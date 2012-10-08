#pragma once
#include "dismember.h"


#if __CUDA_ARCH__ < 300
#include "shfl_sm_10.h"
#else

namespace trove {
namespace detail {

template<int s>
struct shuffle {
    __device__
    static void impl(int d[s], const int& i) {
        d[0] = __shfl(d[0], i);
        shuffle<s-1>::impl(d+1, i);
    }
};

template<>
struct shuffle<1> {
    __device__
    static void impl(int d[1], const int& i) {
        d[0] = __shfl(d[0], i);
    }
};

}
}

template<typename T>
__device__
T __shfl(const T& t, const int& i) {
    union trove::detail::dismember<T,
    trove::detail::size_in_ints<T>::value> cleaver;
    cleaver.d = t;
    trove::detail::shuffle<trove::detail::size_in_ints<T>::value>
      ::impl(cleaver.i, i);
    return cleaver.d;
}

#endif
