#pragma once

namespace trove {
namespace detail {

template<typename T>
struct size_in_ints {
    static const int value = (sizeof(T) - 1)/sizeof(int) + 1;
};

template<typename T, int s>
union dismember {
    T d;
    int i[s];
};


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

