#pragma once
#include <trove/array.h>

namespace trove {
namespace detail {

template<typename T>
struct size_in_ints {
    static const int value = (sizeof(T) - 1)/sizeof(int) + 1;
};

template<typename T, int r=size_in_ints<T>::value>
struct dismember {
    typedef array<int, r> result_type;
    static const int idx = size_in_ints<T>::value - r;
    __host__ __device__
    static result_type impl(const T& t) {
        return result_type(((const int*)&t)[idx],
                           dismember<T, r-1>::impl(t));
    }
};

template<typename T>
struct dismember<T, 1> {
    typedef array<int, 1> result_type;
    static const int idx = size_in_ints<T>::value - 1;
    __host__ __device__
    static result_type impl(const T& t) {
        return result_type(((const int*)&t)[idx]);
    }
};


template<typename T, int r=size_in_ints<T>::value>
struct remember {
    static const int idx = size_in_ints<T>::value - r;
    __host__ __device__
    static void impl(const array<int, r>& d, T& t) {
        ((int*)&t)[idx] = d.head;
        remember<T, r-1>::impl(d.tail, t);
    }
};

template<typename T>
struct remember<T, 1> {
    static const int idx = size_in_ints<T>::value - 1;
    __host__ __device__
    static void impl(const array<int, 1>& d, const T& t) {
        ((int*)&t)[idx] = d.head;
    }
};


template<typename T>
__host__ __device__
array<int, detail::size_in_ints<T>::value> lyse(const T& in) {
    return detail::dismember<T>::impl(in);
}

template<typename T>
__host__ __device__
T fuse(const array<int, detail::size_in_ints<T>::value>& in) {
    T result;
    detail::remember<T>::impl(in, result);
    return result;
}

}
}