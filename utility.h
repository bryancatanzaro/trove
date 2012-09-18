#pragma once
#include <thrust/tuple.h>

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

}
