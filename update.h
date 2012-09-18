#pragma once
#include "utility.h"

namespace tuple_suite {
namespace detail {

template<typename Tuple>
struct update_tuple{};

template<typename HT, typename TT>
struct update_tuple<thrust::detail::cons<HT, TT> > {
    __host__ __device__
    static void impl(thrust::detail::cons<HT, TT>& tup, const HT& d, int idx) {
        if(idx == 0) tup.get_head() = d;
        update_tuple<TT>::impl(tup.get_tail(), d, idx-1);
    }
};

template<>
struct update_tuple<thrust::null_type> {
    template<typename T>
    __host__ __device__
    static void impl(thrust::null_type, T, int) {}
};

}

template<typename Tuple, typename T>
__host__ __device__
void update(Tuple& tup, const T& d, int idx) {
    update_tuple<typename cons_type<Tuple>::type>::impl(tup, d, idx);
}

}
