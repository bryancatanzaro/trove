#pragma once
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>

namespace trove {

/*! \p cons_type is a metafunction that computes the
 * <tt>thrust::detail::cons</tt> type for a <tt>thrust::tuple</tt>
 */
template<typename Tuple>
struct cons_type {
    typedef thrust::detail::cons<
        typename Tuple::head_type,
        typename Tuple::tail_type> type;
};

template<int N, typename T>
struct homogeneous_tuple {
    typedef thrust::detail::cons<
        T, typename homogeneous_tuple<N-1, T>::type> type;
};

template<typename T>
struct homogeneous_tuple<0, T> {
    typedef thrust::null_type type;
};


template<typename T>
struct counting_tuple{};

template<typename HT, typename TT>
struct counting_tuple<thrust::detail::cons<HT, TT> > {
    typedef thrust::detail::cons<HT, TT> Tuple;
    __host__ __device__
    static Tuple impl(HT v=0, HT i=1) {
        return Tuple(v,
                     counting_tuple<TT>::impl(v + i, i));
    }
};

template<>
struct counting_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__
    static thrust::null_type impl(T v=0, T i=1) {
        return thrust::null_type();
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


template<int m, int p=0>
struct greatest_factor_of_two {
    static const int value = (m & 0x1) == 0 ? greatest_factor_of_two<m >> 1, p+1>::value : p;
};

template<int p>
struct greatest_factor_of_two<0, p> {
    static const int value = p;
};

template<int m>
struct is_power_of_two {
    static const bool value = (m & (m-1)) == 0;
};




}
