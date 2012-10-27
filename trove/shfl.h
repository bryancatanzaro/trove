#pragma once
#include <trove/array.h>
#include <trove/detail/dismember.h>


namespace trove {
namespace detail {

template<int s>
struct shuffle {
    __device__
    static void impl(array<int, s>& d, const int& i) {
        d.head = __shfl(d.head, i);
        shuffle<s-1>::impl(d.tail, i);
    }
};

template<>
struct shuffle<1> {
    __device__
    static void impl(array<int, 1>& d, const int& i) {
        d.head = __shfl(d.head, i);
    }
};

}
}

template<typename T>
__device__
T __shfl(const T& t, const int& i) {
    typedef trove::array<int,
                         trove::detail::aliased_size<T, int>::value>
        lysed_array;
    lysed_array lysed = trove::detail::lyse<int>(t);
    trove::detail::shuffle<lysed_array::size>
      ::impl(lysed, i);
    return trove::detail::fuse<T>(lysed);
}
