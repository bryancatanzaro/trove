#pragma once
#include <trove/array.h>

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
struct counting_array<array<T, 1> > {
    __host__ __device__
    static array<T, 1> impl(T v, T) {
        return make_array(v);
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

template<bool cond, typename T, typename Then, typename Else>
struct value_if {
    static const T value = Then::value;
};

template<typename T, typename Then, typename Else>
struct value_if<false, T, Then, Else> {
    static const T value = Else::value;
};

template<typename T, T x>
struct value_identity {
    static const T value = x;
};

template<typename T, template<T> class Fn, T x, T p=0>
struct inverse {
    static const T value =
        value_if<Fn<p>::value == x, T,
                 value_identity<T, p>, inverse<T, Fn, x, p+1> >::value;
};

struct null_type{};

template<typename T, T i, typename Tail=null_type>
struct cons_c {
    static const T head = i;
    typedef Tail tail;
};

template<int k, int l>
struct static_range {
    static const int head = k;
    typedef static_range<k+1, l> tail;
};

template<int f>
struct static_range<f, f> {
    static const int head = f;
    typedef null_type tail;
};


}
