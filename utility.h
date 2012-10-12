#pragma once
#include "array.h"

namespace trove {

template<typename T>
struct counting_array{};

template<typename T, int s>
struct counting_array<array<T, s> > {
    typedef array<T, s> Array;
    __host__ __device__
    static Array impl(T v=0, T i=1) {
        return Array(v,
                     counting_array<array<T, s-1> >::impl(v + i, i));
    }
};

template<typename T>
struct counting_array<array<T, 0> > {
    __host__ __device__
    static array<T, 0> impl(T, T) {
        return make_array<T>();
    }
};

template<int m>
struct static_log {
    static const int value = 1 + static_log<m >> 1>::value;
};

template<>
struct static_log<1> {
    static const int value = 0;
};

template<>
struct static_log<0> {
    //This functions as a static assertion
    //Don't take the log of 0!!
};

template<int m>
struct is_power_of_two {
    static const bool value = (m & (m-1)) == 0;
};

template<int m>
struct is_odd {
    static const bool value = (m & 1) == 1;
};


}
