#pragma once

namespace trove {

template<typename T, int m>
struct array {
    typedef T value_type;
    typedef T head_type;
    typedef array<T, m-1> tail_type;
    static const int size = m;
    head_type head;
    tail_type tail;
    __host__ __device__
    array(head_type h, const tail_type& t) : head(h), tail(t) {}
};

template<typename T>
struct array<T, 0> {};

namespace detail {

template<typename T, int m, int i>
struct get_impl {
    __host__ __device__ static T& impl(array<T, m>& src) {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
    __host__ __device__ static T impl(const array<T, m>& src) {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
};

template<typename T, int m>
struct get_impl<T, m, 0> {
    __host__ __device__ static T& impl(array<T, m>& src) {
        return src.head;
    }
    __host__ __device__ static T impl(const array<T, m>& src) {
        return src.head;
    }
};

}

template<int i, typename T, int m>
__host__ __device__
T& get(array<T, m>& src) {
    return detail::get_impl<T, m, i>::impl(src);
}

template<int i, typename T, int m>
__host__ __device__
T get(const array<T, m>& src) {
    return detail::get_impl<T, m, i>::impl(src);
}

template<typename T>
__host__ __device__
array<T, 0> make_array() {
    return array<T, 0>();
}

template<typename T>
__host__ __device__
array<T, 1> make_array(T a0) {
    return array<T, 1>(a0, make_array<T>());
}

template<typename T>
__host__ __device__
array<T, 2> make_array(T a0, T a1) {
    return array<T, 2>(a0,
                       make_array<T>(a1));
}

template<typename T>
__host__ __device__
array<T, 3> make_array(T a0, T a1, T a2) {
    return array<T, 3>(a0,
                       make_array<T>(a1, a2));
}

template<typename T>
__host__ __device__
array<T, 4> make_array(T a0, T a1, T a2, T a3) {
    return array<T, 4>(a0,
                       make_array<T>(a1, a2, a3));
}

template<typename T>
__host__ __device__
array<T, 5> make_array(T a0, T a1, T a2, T a3, T a4) {
    return array<T, 5>(a0,
                       make_array<T>(a1, a2, a3, a4));
}

template<typename T>
__host__ __device__
array<T, 6> make_array(T a0, T a1, T a2, T a3, T a4,
                       T a5) {
    return array<T, 6>(a0,
                       make_array<T>(a1, a2, a3, a4, a5));
}

template<typename T>
__host__ __device__
array<T, 7> make_array(T a0, T a1, T a2, T a3, T a4,
                       T a5, T a6) {
    return array<T, 7>(a0,
                       make_array<T>(a1, a2, a3, a4, a5,
                                     a6));
}

template<typename T>
__host__ __device__
array<T, 8> make_array(T a0, T a1, T a2, T a3, T a4,
                       T a5, T a6, T a7) {
    return array<T, 8>(a0,
                       make_array<T>(a1, a2, a3, a4, a5,
                                     a6, a7));
}

template<typename T>
__host__ __device__
array<T, 9> make_array(T a0, T a1, T a2, T a3, T a4,
                       T a5, T a6, T a7, T a8) {
    return array<T, 9>(a0,
                       make_array<T>(a1, a2, a3, a4, a5,
                                     a6, a7, a8));
}

template<typename T>
__host__ __device__
array<T, 10> make_array(T a0, T a1, T a2, T a3, T a4,
                        T a5, T a6, T a7, T a8, T a9) {
    return array<T, 10>(a0,
                        make_array<T>(a1, a2, a3, a4, a5,
                                      a6, a7, a8, a9));
}


namespace detail {

template<typename T, int s>
struct make_array_impl {
    typedef typename array<T, s>::type result_type;
    __host__ __device__
    static result_type impl(T ary[s]) {
        return result_type(ary[0],
                           make_array_impl<T, s-1>::impl(ary+1));
    }
};

template<typename T>
struct make_array_impl<T, 0> {
    __host__ __device__
    static array<T, 0> impl(T ary[0]) {
        return array<T, 0>();
    }
};

template<typename T, int s>
struct make_carray_impl {
    typedef array<T, s> array_type;
    __host__ __device__
    static void impl(const array_type& ary, T result[s]) {
        result[0] = ary.head;
        make_carray_impl<T, s-1>::impl(ary.tail, result+1);
    }
};

template<typename T>
struct make_carray_impl<T, 0> {
    __host__ __device__
    static void impl(array<T, 0>, T result[0]) {}
};
 
} //end namespace detail
 
template<typename T, int s>
__host__ __device__
array<T, s> make_array(T cary[s]) {
    return detail::make_array_impl<T, s>::impl(cary);
}

template<typename T, int s>
__host__ __device__
void make_carray(const array<T, s>& ary,
                 T result[s]) {
    detail::make_carray_impl<T, s>::impl(ary, result);
}
  
} //end namespace trove
