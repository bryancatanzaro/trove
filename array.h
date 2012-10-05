#pragma once
#include "utility.h"

namespace trove {

namespace detail {

template<typename T, int s>
struct make_tuple_impl {
    typedef typename homogeneous_tuple<s, T>::type result_type;
    __host__ __device__
    static result_type impl(T ary[s]) {
        return result_type(ary[0],
                           make_tuple_impl<T, s-1>::impl(ary+1));
    }
};

template<typename T>
struct make_tuple_impl<T, 0> {
    __host__ __device__
    static thrust::null_type impl(T ary[0]) {
        return thrust::null_type();
    }
};

template<typename T, int s>
struct make_array_impl {
    typedef typename homogeneous_tuple<s, T>::type cns_type;
    __host__ __device__
    static void impl(const cns_type& cns, T result[s]) {
        result[0] = cns.get_head();
        make_array_impl<T, s-1>::impl(cns.get_tail(), result+1);
    }
};

template<typename T>
struct make_array_impl<T, 0> {
    __host__ __device__
    static void impl(const thrust::null_type&, T ary[0]) {}
};
 
} //end namespace detail
 
template<typename T, int s>
__host__ __device__
typename homogeneous_tuple<s, T>::type make_tuple(T ary[s]) {
    return detail::make_tuple_impl<T, s>::impl(ary);
}

template<typename T, int s>
__host__ __device__
void make_array(const typename homogeneous_tuple<s, T>::type& cns,
                T result[s]) {
    detail::make_array_impl<T, s>::impl(cns, result);
}
  
} //end namespace trove
