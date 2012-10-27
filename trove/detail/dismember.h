#pragma once
#include <trove/array.h>
#include <trove/utility.h>

namespace trove {
namespace detail {


template<typename T,
         bool use_int=size_multiple_power_of_two<T, 2>::value,
         bool use_int2=size_multiple_power_of_two<T, 3>::value,
         bool use_int4=size_multiple_power_of_two<T, 4>::value >
struct dismember_type {
    typedef char type;
};

template<typename T>
struct dismember_type<T, true, false, false> {
    typedef int type;
};

template<typename T>
struct dismember_type<T, true, true, false> {
    typedef int2 type;
};

template<typename T>
struct dismember_type<T, true, true, true> {
    typedef int4 type;
};


template<typename T, typename U>
struct aliased_size {
    static const int value = (sizeof(T) - 1)/sizeof(U) + 1;
};

template<typename T,
         typename U=typename dismember_type<T>::type,
         int r=aliased_size<T, U>::value>
struct dismember {
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__
    static result_type impl(const T& t) {
        return result_type(((const U*)&t)[idx],
                           dismember<T, U, r-1>::impl(t));
    }
};

template<typename T, typename U>
struct dismember<T, U, 1> {
    typedef array<U, 1> result_type;
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__
    static result_type impl(const T& t) {
        return result_type(((const U*)&t)[idx]);
    }
};


template<typename T,
         typename U=typename dismember_type<T>::type,
         int r=aliased_size<T, U>::value>
struct remember {
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__
    static void impl(const array<U, r>& d, T& t) {
        ((U*)&t)[idx] = d.head;
        remember<T, U, r-1>::impl(d.tail, t);
    }
};

template<typename T, typename U>
struct remember<T, U, 1> {
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__
    static void impl(const array<U, 1>& d, const T& t) {
        ((U*)&t)[idx] = d.head;
    }
};


template<typename U, typename T>
__host__ __device__
array<U, detail::aliased_size<T, U>::value> lyse(const T& in) {
    return detail::dismember<T, U>::impl(in);
}

template<typename T, typename U>
__host__ __device__
T fuse(const array<U, detail::aliased_size<T, U>::value>& in) {
    T result;
    detail::remember<T, U>::impl(in, result);
    return result;
}

}
}
