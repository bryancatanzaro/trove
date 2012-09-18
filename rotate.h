#pragma once
#include <thrust/tuple.h>

namespace trove {
namespace detail {

template<typename Tuple, int i, int j=0>
struct rotate_elements;

template<typename Tuple, int i, int j, bool non_terminal>
struct rotate_elements_helper {
    static const int size = thrust::tuple_size<Tuple>::value;
    static const int other = (i + j) % size;
    static const bool new_non_terminal = j < size-2;
    __host__ __device__
    static void impl(const Tuple& t, int a, Tuple& r) {
        if (a & i)
            thrust::get<j>(r) = thrust::get<other>(t);
        rotate_elements_helper<Tuple, i, j+1, new_non_terminal>::impl(t, a, r);
    }
};

template<typename Tuple, int i, int j>
struct rotate_elements_helper<Tuple, i, j, false> {
    static const int size = thrust::tuple_size<Tuple>::value;
    static const int other = (i + j) % size;
    __host__ __device__
    static void impl(const Tuple& t, int a, Tuple& r) {
        if (a & i)
            thrust::get<j>(r) = thrust::get<other>(t);
    }
};


template<typename Tuple, int i, int j>
struct rotate_elements{
    static const int size = thrust::tuple_size<Tuple>::value;
    static const bool non_terminal = j < size-1;
    __host__ __device__
    static void impl(const Tuple& t, int a, Tuple& r) {
        rotate_elements_helper<Tuple, i, 0, non_terminal>::impl(t, a, r);
    }
};

template<typename Tuple, int i>
struct rotate_impl;

template<typename Tuple, int i, bool non_terminal>
struct rotate_impl_helper {
    static const int size = thrust::tuple_size<Tuple>::value;
    static const int next_i = i * 2;
    __host__ __device__
    static Tuple impl(const Tuple& t, int a) {
        Tuple rotated = t;
        rotate_elements<Tuple, i>::impl(t, a, rotated);
        return rotate_impl<Tuple, next_i>::impl(rotated, a);
    }
};

template<typename Tuple, int i>
struct rotate_impl_helper<Tuple, i, false> {
    static const int size = thrust::tuple_size<Tuple>::value;
    __host__ __device__
    static Tuple impl(const Tuple& t, int a) {
        Tuple rotated = t;
        rotate_elements<Tuple, i>::impl(t, a, rotated);
        return rotated;
    }
};
    
template<typename Tuple, int i>
struct rotate_impl {
    static const int size = thrust::tuple_size<Tuple>::value;
    static const int next_i = i * 2;
    static const bool non_terminal = next_i < size;
    __host__ __device__
    static Tuple impl(const Tuple& t, int a) {
        return rotate_impl_helper<Tuple, i, non_terminal>::impl(t, a);
    }
};

} //ends namespace detail

template<typename Tuple>
__host__ __device__
Tuple rotate(const Tuple& t, int a) {
    return detail::rotate_impl<Tuple, 1>::impl(t, a);
}

} //ends namespace trove
