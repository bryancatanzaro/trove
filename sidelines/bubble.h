#pragma once
#include <thrust/tuple.h>
#include <thrust/swap.h>
#include <iostream>

namespace trove {
namespace detail {

template<typename Key, typename Value, int top_index, int bottom_index>
struct bubble_sbk_step;

template<typename Key, typename Value, bool non_terminal,
         int top_index, int bottom_index>
struct bubble_sbk_swap {
    __host__ __device__ static void impl(Key& k, Value& v) {
        if(thrust::get<top_index>(k) <
           thrust::get<top_index-1>(k)) 
            thrust::swap(thrust::get<top_index>(v),
                         thrust::get<top_index-1>(v));
        if(thrust::get<top_index>(k) <
           thrust::get<top_index-1>(k)) 
            thrust::swap(thrust::get<top_index>(k),
                         thrust::get<top_index-1>(k));

        bubble_sbk_step<Key, Value, top_index-1,
                        bottom_index>::impl(k, v);
    }
};
    

template<typename Key, typename Value,
         int top_index, int bottom_index>
struct bubble_sbk_swap<Key, Value, false, top_index, bottom_index> {
    __host__ __device__ static void impl(Key& k, Value& v) {
        if(thrust::get<top_index>(k) <
           thrust::get<top_index-1>(k)) 
            thrust::swap(thrust::get<top_index>(v),
                         thrust::get<top_index-1>(v));
        if(thrust::get<top_index>(k) <
           thrust::get<top_index-1>(k)) 
            thrust::swap(thrust::get<top_index>(k),
                         thrust::get<top_index-1>(k));

    }
};
    
template<typename Key, typename Value, int top_index, int bottom_index>
struct bubble_sbk_step {
    static const bool non_terminal = top_index != bottom_index;

    __host__ __device__ static void impl(Key& k, Value& v) {
        bubble_sbk_swap<Key, Value, non_terminal, top_index, bottom_index>
            ::impl(k, v);
    }
};

template<typename Key, typename Value, int iteration, int size>
struct bubble_sbk_level;

template<int iteration, int size>
struct calculate_levels {
    static const int top_level = size;
    static const int bottom_level = iteration + 1;
};

template<typename Key, typename Value,
         bool non_terminal, int iteration, int size>
struct bubble_sbk_level_helper {
    typedef calculate_levels<iteration, size> calculator;
 
    __host__ __device__ static void impl(Key& k, Value& v) {
        bubble_sbk_step<Key, Value, calculator::top_level,
                        calculator::bottom_level>::impl(k, v);
        bubble_sbk_level<Key, Value, iteration+1, size>::impl(k, v);
    }  
};

template<typename Key, typename Value,
         int iteration, int size>
struct bubble_sbk_level_helper<Key, Value, false, iteration, size> {
    typedef calculate_levels<iteration, size> calculator;
    
    __host__ __device__ static void impl(Key& k, Value& v) {
        bubble_sbk_step<Key, Value, calculator::top_level,
                        calculator::bottom_level>::impl(k, v);
    }
};

template<typename Key, typename Value, int iteration, int size>
struct bubble_sbk_level {
    static const bool non_terminal = iteration < size - 1;

    __host__ __device__ static void impl(Key& k, Value& v) {
        bubble_sbk_level_helper<Key, Value, non_terminal,
                                iteration, size>::impl(k, v);
    }
};

} //ends namespace detail

/*! Sorts a key, value tuple pair using bubble sort */
template<typename Key, typename Value>
__host__ __device__
void bubble_sort_by_key(Key& k, Value& v) {
    detail::bubble_sbk_level<Key, Value, 0, thrust::tuple_size<Key>::value-1>
        ::impl(k, v);
};

} //ends namespace trove
