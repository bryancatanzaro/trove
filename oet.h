#pragma once
#include <thrust/tuple.h>
#include <thrust/swap.h>
#include <iostream>

template<typename Key, typename Value, int top_index, int bottom_index>
struct oet_sbk_step;

template<typename Key, typename Value, bool non_terminal,
         int offset, int size>
struct oet_sbk_swap {
    __host__ __device__ static void impl(Key& k, Value& v) {
        if(thrust::get<offset + 1>(k) <
           thrust::get<offset>(k)) 
            thrust::swap(thrust::get<offset + 1>(v),
                         thrust::get<offset>(v));
        if(thrust::get<offset + 1>(k) <
           thrust::get<offset>(k)) 
            thrust::swap(thrust::get<offset + 1>(k),
                         thrust::get<offset>(k));

        oet_sbk_step<Key, Value, offset + 2,
                        size>::impl(k, v);
    }
};
    

template<typename Key, typename Value,
         int offset, int size>
struct oet_sbk_swap<Key, Value, false, offset, size> {
    __host__ __device__ static void impl(Key& k, Value& v) {
        if(thrust::get<offset + 1>(k) <
           thrust::get<offset>(k)) 
            thrust::swap(thrust::get<offset + 1>(v),
                         thrust::get<offset>(v));
        if(thrust::get<offset + 1>(k) <
           thrust::get<offset>(k)) 
            thrust::swap(thrust::get<offset + 1>(k),
                         thrust::get<offset>(k));

    }
};
    
template<typename Key, typename Value, int offset, int size>
struct oet_sbk_step {
    static const bool non_terminal = offset + 2 < size;

    __host__ __device__ static void impl(Key& k, Value& v) {
        oet_sbk_swap<Key, Value, non_terminal, offset, size>
            ::impl(k, v);
    }
};

template<typename Key, typename Value, int iteration, int size>
struct oet_sbk_level;


template<typename Key, typename Value,
         bool non_terminal, int iteration, int size>
struct oet_sbk_level_helper {
 
    __host__ __device__ static void impl(Key& k, Value& v) {
        oet_sbk_step<Key, Value, iteration & 0x1, size>::impl(k, v);
        oet_sbk_level<Key, Value, iteration+1, size>::impl(k, v);
    }  
};

template<typename Key, typename Value,
         int iteration, int size>
struct oet_sbk_level_helper<Key, Value, false, iteration, size> {
    
    __host__ __device__ static void impl(Key& k, Value& v) {
        oet_sbk_step<Key, Value, iteration & 0x1, size>::impl(k, v);
    }
};

template<typename Key, typename Value, int iteration, int size>
struct oet_sbk_level {
    static const bool non_terminal = iteration < size;

    __host__ __device__ static void impl(Key& k, Value& v) {
        oet_sbk_level_helper<Key, Value, non_terminal,
                                iteration, size>::impl(k, v);
    }
};

/*! Sorts a key, value tuple pair */
template<typename Key, typename Value>
__host__ __device__
void oet_sort_by_key(Key& k, Value& v) {
    oet_sbk_level<Key, Value, 0, thrust::tuple_size<Key>::value-1>
        ::impl(k, v);
};

