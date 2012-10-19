#pragma once
#include <trove/array.h>

namespace trove {
namespace detail {

template<typename Array, int i, int j=0>
struct rotate_elements;

template<typename Array, int i, int j, bool non_terminal>
struct rotate_elements_helper {
    static const int size = Array::size;
    static const int other = (i + j) % size;
    static const bool new_non_terminal = j < size-2;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        if (a & i)
            trove::get<j>(r) = trove::get<other>(t);
        rotate_elements_helper<Array, i, j+1, new_non_terminal>::impl(t, a, r);
    }
};

template<typename Array, int i, int j>
struct rotate_elements_helper<Array, i, j, false> {
    static const int size = Array::size;
    static const int other = (i + j) % size;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        if (a & i)
            trove::get<j>(r) = trove::get<other>(t);
    }
};


template<typename Array, int i, int j>
struct rotate_elements{
    static const int size = Array::size;
    static const bool non_terminal = j < size-1;
    __host__ __device__
    static void impl(const Array& t, int a, Array& r) {
        rotate_elements_helper<Array, i, 0, non_terminal>::impl(t, a, r);
    }
};

template<typename Array, int i>
struct rotate_impl;

template<typename Array, int i, bool non_terminal>
struct rotate_impl_helper {
    static const int size = Array::size;
    static const int next_i = i * 2;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        Array rotated = t;
        rotate_elements<Array, i>::impl(t, a, rotated);
        return rotate_impl<Array, next_i>::impl(rotated, a);
    }
};

template<typename Array, int i>
struct rotate_impl_helper<Array, i, false> {
    static const int size = Array::size;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        Array rotated = t;
        rotate_elements<Array, i>::impl(t, a, rotated);
        return rotated;
    }
};
    
template<typename Array, int i>
struct rotate_impl {
    static const int size = Array::size;
    static const int next_i = i * 2;
    static const bool non_terminal = next_i < size;
    __host__ __device__
    static Array impl(const Array& t, int a) {
        return rotate_impl_helper<Array, i, non_terminal>::impl(t, a);
    }
};

} //ends namespace detail

template<typename T, int i>
__host__ __device__
array<T, i> rotate(const array<T, i>& t, int a) {
    return detail::rotate_impl<array<T, i>, 1>::impl(t, a);
}

} //ends namespace trove
