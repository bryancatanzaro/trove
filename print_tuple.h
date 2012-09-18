#pragma once
#include <iostream>

namespace trove {
namespace detail {

template<typename T>
struct print_tuple_helper{};

template<typename HT, typename TT>
struct print_tuple_helper<thrust::detail::cons<HT, TT> > {
    static void impl(const thrust::detail::cons<HT, TT>& a) {
        std::cout << a.get_head() << " ";
        print_tuple_helper<TT>::impl(a.get_tail());
    }
};

template<>
struct print_tuple_helper<thrust::null_type> {
    static void impl(thrust::null_type) {
        std::cout << std::endl;
    }
};

} //ends namespace detail

template<typename Tuple>
__host__
void print_tuple(const Tuple& a) {
    print_tuple_helper<typename cons_type<Tuple>::type>::impl(a);
}

} //ends namespace trove
