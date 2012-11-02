#pragma once
#include <trove/detail/dismember.h>

namespace trove {
namespace detail {

template<int s, typename T>
struct divergent_loader {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, s> impl(const U* src) {
        return array<U, s>(*src,
                           divergent_loader<s-1, T>::impl(src+1));
    }
    __device__
    static array<U, s> impl(const T* src) {
        return impl((U*)src);
    }
};

template<typename T>
struct divergent_loader<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static array<U, 1> impl(const U* src) {
        return array<U, 1>(*src);
    }
    __device__
    static array<U, 1> impl(const T* src) {
        return impl((U*)src);
    }
};


template<typename T>
__device__
T divergent_load(const T* src) {
    typedef typename detail::dismember_type<T>::type U;
    typedef array<U, detail::aliased_size<T, U>::value> u_store;
    u_store loaded =
        detail::divergent_loader<detail::aliased_size<T, U>::value, T>::impl(
            src);
    return detail::fuse<T>(loaded);
}

template<int s, typename T>
struct divergent_storer {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, s>& data, U* dest) {
        *dest = data.head;
        divergent_storer<s-1, T>::impl(data.tail, dest+1);
    }
    __device__
    static void impl(const array<U, s>& data, const T* dest) {
        return impl(data, (U*)dest);
    }
};

template<typename T>
struct divergent_storer<1, T> {
    typedef typename detail::dismember_type<T>::type U;
    __device__
    static void impl(const array<U, 1>& data, U* dest) {
        *dest = data.head;
    }
    __device__
    static void impl(const array<U, 1>& data, const T* dest) {
        return impl(data, (U*)dest);
    }
};

template<typename T>
__device__
void divergent_store(const T& data, T* dest) {
    typedef typename detail::dismember_type<T>::type U;
    typedef array<U, detail::aliased_size<T, U>::value> u_store;
    u_store lysed = detail::lyse<U>(data);
    detail::divergent_storer<detail::aliased_size<T, U>::value, T>::impl(
        lysed, dest);
}


}
}
