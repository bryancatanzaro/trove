#pragma once
#include <trove/aos.h>

namespace trove {
namespace detail {

template<typename T>
struct coalesced_ref {
    T* m_ptr;
    __device__ explicit coalesced_ref(T* ptr) : m_ptr(ptr) {}
    
    __device__ operator T() {
        return trove::load(m_ptr);
    }
    __device__ coalesced_ref& operator=(const T& data) {
        trove::store(data, m_ptr);
        return *this;
    }

    __device__ coalesced_ref& operator=(const coalesced_ref& other) {
        T data = trove::load(other.m_ptr);
        trove::store(data, m_ptr);
        return *this;
    }
};
}

template<typename T>
struct coalesced_ptr {
    T* m_ptr;
    __device__ coalesced_ptr(T* ptr) : m_ptr(ptr) {}
    __device__ trove::detail::coalesced_ref<T> operator*() {
        return trove::detail::coalesced_ref<T>(m_ptr);
    }
    template<typename I>
    __device__ trove::detail::coalesced_ref<T> operator[](const I& idx) {
        return trove::detail::coalesced_ref<T>(m_ptr + idx);
    }
    __device__ operator T*() {
        return m_ptr;
    }
};





}
