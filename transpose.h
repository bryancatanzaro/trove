#pragma once
#include "utility.h"
#include "rotate.h"
#include "shfl.h"
#include "static_mod_inverse.h"
#include "static_gcd.h"

#define WARP_SIZE 32
#define WARP_MASK 0x1f
#define LOG_WARP_SIZE 5

namespace trove {
namespace detail {

struct odd{};
struct power_of_two{};
struct composite{};

template<int m, bool ispo2=is_power_of_two<m>::value, bool isodd=is_odd<m>::value>
struct tx_algorithm {
    typedef composite type;
};

template<int m>
struct tx_algorithm<m, true, false> {
    typedef power_of_two type;
};

template<int m>
struct tx_algorithm<m, false, true> {
    typedef odd type;
};

template<int m, typename Schema=typename tx_algorithm<m>::type>
struct c2r_offset_constants{};

template<int m>
struct c2r_offset_constants<m, odd> {
    static const int offset = WARP_SIZE - static_mod_inverse<m, WARP_SIZE>::value;
    static const int rotate = static_mod_inverse<WARP_SIZE, m>::value;
    static const int permute = static_mod_inverse<rotate, m>::value;
};

template<int m>
struct c2r_offset_constants<m, power_of_two> {
    static const int offset = WARP_SIZE - WARP_SIZE/m; 
    static const int permute = m - 1;
};

template<int m>
struct c2r_offset_constants<m, composite> {
    static const int c = static_gcd<m, WARP_SIZE>::value;
    static const int k = static_mod_inverse<m/c, WARP_SIZE/c>::value;
};

template<int m, typename Schema=typename tx_algorithm<m>::type>
struct r2c_offset_constants{};

template<int m>
struct r2c_offset_constants<m, odd> {
    static const int permute = static_mod_inverse<WARP_SIZE, m>::value;
};

template<int m>
struct r2c_offset_constants<m, power_of_two> :
        c2r_offset_constants<m, power_of_two> {
};

template<typename T, template<int> class Permute, int position=0>
struct tx_permute_impl{};

template<typename T, int s, template<int> class Permute, int position>
struct tx_permute_impl<array<T, s>, Permute, position> {
    typedef array<T, s> Remaining;
    static const int idx = Permute<position>::value;
    template<typename Source>
    __host__ __device__
    static Remaining impl(const Source& src) {
        return Remaining(
            trove::get<idx>(src),
            tx_permute_impl<array<T, s-1>, Permute, position+1>::impl(
                src));
    }
};

template<typename T, template<int> class Permute, int position>
struct tx_permute_impl<array<T, 0>, Permute, position> {
    template<typename Source>
    __host__ __device__
    static array<T, 0> impl(const Source&) {
        return array<T, 0>();
    }
};


template<int m, int a, int b=0>
struct affine_modular_fn {
    template<int x>
    struct eval {
        static const int value = (a * x + b) % m;
    };
};

template<int m>
struct composite_c2r_permute_fn {
    static const int o = WARP_SIZE % m;
    static const int c = static_gcd<m, WARP_SIZE>::value;
    static const int p = m / c;
    template<int x>
    struct eval {
        static const int value = (x * o - (x / p)) % m;
    };
};




template<typename Array>
__host__ __device__ Array c2r_tx_permute(const Array& t) {
    return tx_permute_impl<
        Array,
        affine_modular_fn<Array::size,
        c2r_offset_constants<Array::size>::permute>::template eval>::impl(t);
}



template<typename Array>
__host__ __device__ Array composite_c2r_tx_permute(const Array& t) {
    return tx_permute_impl<
        Array,
        composite_c2r_permute_fn<Array::size>::template eval>::impl(t);
}


template<typename Array>
__host__ __device__ Array r2c_tx_permute(const Array& t) {
    return tx_permute_impl<
        Array,
        affine_modular_fn<Array::size,
                          r2c_offset_constants<Array::size>::permute>::template eval>::impl(t);
}


template<typename Array, int b, int o>
struct c2r_compute_offsets_impl{};

template<int s, int b, int o>
struct c2r_compute_offsets_impl<array<int, s>, b, o> {
    typedef array<int, s> Array;
    __device__
    static Array impl(int offset) {
        if (offset >= b) {
            offset -= b;
        } //Poor man's x % b. Requires that o < b.
        return Array(offset,
                     c2r_compute_offsets_impl<array<int, s-1>, b, o>::
                     impl(offset + o));
    }
};

template<int b, int o>
struct c2r_compute_offsets_impl<array<int, 0>, b, o> {
    __device__
    static array<int, 0> impl(int) {
        return array<int, 0>();
    }
};

template<int m, typename Schema>
struct c2r_compute_initial_offset {};

template<int m>
struct c2r_compute_initial_offset<m, odd> {
    typedef c2r_offset_constants<m> constants;
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((WARP_SIZE - warp_id) * constants::offset)
            & WARP_MASK;
        return initial_offset;
    }
};

template<int m>
struct c2r_compute_initial_offset<m, power_of_two> {
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = ((warp_id * (WARP_SIZE + 1)) >>
                              static_log<m>::value)
            & WARP_MASK;
        return initial_offset;
    }
};

template<int m, typename Schema>
struct r2c_compute_initial_offset {};

template<int m>
struct r2c_compute_initial_offset<m, odd> {
    __device__ static int impl() {
        int warp_id = threadIdx.x & WARP_MASK;
        int initial_offset = (warp_id * m) & WARP_MASK;
        return initial_offset;
    }
};


template<int m, typename Schema>
__device__
array<int, m> c2r_compute_offsets() {
    typedef c2r_offset_constants<m> constants;
    int initial_offset = c2r_compute_initial_offset<m, Schema>::impl();
    return c2r_compute_offsets_impl<array<int, m>,
                                    WARP_SIZE,
                                    constants::offset>::impl(initial_offset);
}

template<typename T, int m, int p = 0>
struct c2r_compute_composite_offsets{};
 
template<int s, int m, int p>
struct c2r_compute_composite_offsets<array<int, s>, m, p> {
    static const int n = WARP_SIZE;
    static const int mod_n = n - 1;
    static const int c = static_gcd<m, WARP_SIZE>::value;
    static const int k = static_mod_inverse<m/c, n/c>::value;
    static const int mod_c = c - 1;
    static const int log_c = static_log<c>::value;
    static const int n_div_c = n / c;
    static const int mod_n_div_c = n_div_c - 1;
    static const int log_n_div_c = static_log<n_div_c>::value;
    typedef array<int, s> result_type;
    __host__ __device__ static result_type impl(int idx, int col) {
        int offset = ((((idx >> log_c) * k) & mod_n_div_c) +
                      ((idx & mod_c) << log_n_div_c)) & mod_n;
        int new_idx = idx + n - 1;
        new_idx = (p == m - c + (col & mod_c)) ? new_idx + m : new_idx;
        return
            result_type(offset,
                        c2r_compute_composite_offsets<array<int, s-1>, m, p+1>
            ::impl(new_idx, col));
                           
    }
};

template<int m, int p>
struct c2r_compute_composite_offsets<array<int, 0>, m, p> {
    __host__ __device__ static array<int, 0> impl(int, int) {
        return array<int, 0>();
    }
};

 
template<int index, int offset, int bound>
struct r2c_offsets {
    static const int value = (offset * index) % bound;
};

template<typename Array, int index, int m, typename Schema>
struct r2c_compute_offsets_impl{};

template<int s, int index, int m>
struct r2c_compute_offsets_impl<array<int, s>, index, m, odd> {
    typedef array<int, s> Array;
    static const int offset = (WARP_SIZE % m * index) % m;
    __device__
    static Array impl(int initial_offset) {
        int current_offset = (initial_offset + offset) & WARP_MASK;
        return Array(current_offset,
                     r2c_compute_offsets_impl<array<int, s-1>,
                     index + 1, m, odd>::impl(initial_offset));
    }
};

template<int index, int m>
struct r2c_compute_offsets_impl<array<int, 0>, index, m, odd> {
    __device__
    static array<int, 0> impl(int) {
        return array<int, 0>();
    }
};


template<int s, int index, int m>
struct r2c_compute_offsets_impl<array<int, s>, index, m, power_of_two> {
    typedef array<int, s> Array;
    __device__
    static Array impl(int initial_offset) {
        int warp_id = threadIdx.x & WARP_MASK;
    
        const int logL = static_log<m>::value;
        const int logP = LOG_WARP_SIZE;
        int msb_bits = warp_id >> (logP - logL);
        int lsb_bits = warp_id & ((1 << (logP - logL)) - 1);
    
        return Array((lsb_bits << logL) | ((msb_bits + m - index) % m),
                     r2c_compute_offsets_impl<array<int, s-1>, index + 1, m, power_of_two>::impl(initial_offset));
    }
};

template<int index, int m>
struct r2c_compute_offsets_impl<array<int, 0>, index, m, power_of_two> {
    __device__
    static array<int, 0> impl(int) {
        return array<int, 0>();
    }
};


template<int m, typename Schema>
__device__
array<int, m> r2c_compute_offsets() {
    typedef r2c_offset_constants<m> constants;
    typedef array<int, m> result_type;
    int initial_offset = r2c_compute_initial_offset<m, Schema>::impl();
    return r2c_compute_offsets_impl<result_type,
                                    0, m, Schema>::impl(initial_offset);
}
        
    
template<typename Data, typename Indices>
struct warp_shuffle {};

template<typename T, int m>
struct warp_shuffle<array<T, m>, array<int, m> > {
    __device__ static void impl(array<T, m>& d,
                                const array<int, m>& i) {
        d.head = __shfl(d.head, i.head);
        warp_shuffle<array<T, m-1>, array<int, m-1> >::impl(d.tail,
                                                            i.tail);
    }
};

template<typename T>
struct warp_shuffle<array<T, 0>, array<int, 0> > {
    __device__ static void impl(array<T, 0>, array<int, 0>) {}
};


template<typename Array, typename Schema>
struct c2r_compute_indices_impl {};

template<typename Array>
struct c2r_compute_indices_impl<Array, odd> {
    __device__ static void impl(Array& indices, int& rotation) {
        indices = detail::c2r_compute_offsets<Array::size, odd>();
        int warp_id = threadIdx.x & WARP_MASK;
        int size = Array::size;
        int r = detail::c2r_offset_constants<Array::size>::rotate;
        rotation = (warp_id * r) % size;
    }
};

template<typename Array>
struct c2r_compute_indices_impl<Array, power_of_two> {
    __device__ static void impl(Array& indices, int& rotation) {
        indices = detail::c2r_compute_offsets<Array::size, power_of_two>();
        int warp_id = threadIdx.x & WARP_MASK;
        int size = Array::size;
        rotation = (size - warp_id) & (size - 1);
    }
};

template<typename Array>
struct c2r_compute_indices_impl<Array, composite> {
    __device__ static void impl(Array& indices, int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
  
        indices = detail::c2r_compute_composite_offsets<Array, Array::size>::
            impl(warp_id, warp_id);
        rotation = warp_id % Array::size;
    }
};

template<typename Array, typename Indices, typename Schema>
struct c2r_warp_transpose_impl {};

template<typename Array, typename Indices>
struct c2r_warp_transpose_impl<Array, Indices, odd> {
    __device__ static void impl(Array& src,
                                const Indices& indices,
                                const int& rotation) {
        detail::warp_shuffle<Array, Indices>::impl(src, indices);
        src = rotate(detail::c2r_tx_permute(src), rotation);
    }
};

template<typename Array, typename Indices>
struct c2r_warp_transpose_impl<Array, Indices, power_of_two> {
    __device__ static void impl(Array& src,
                                const Indices& indices,
                                const int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
        int pre_rotation = warp_id >>
            (LOG_WARP_SIZE -
             static_log<Array::size>::value);
        src = rotate(src, pre_rotation);        
        c2r_warp_transpose_impl<Array, Indices, odd>::impl
            (src, indices, rotation);
    }
};

template<typename Array, typename Indices>
struct c2r_warp_transpose_impl<Array, Indices, composite> {
    __device__ static void impl(Array& src,
                                const Indices& indices,
                                const int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
        int pre_rotation = warp_id >> static_log<WARP_SIZE/static_gcd<Array::size, WARP_SIZE>::value>::value;
        src = rotate(src, pre_rotation);
        detail::warp_shuffle<Array, Indices>::impl(src, indices);
        src = rotate(src, rotation);
        src = composite_c2r_tx_permute(src);
    }
};

template<typename Array, typename Schema>
struct r2c_compute_indices_impl {};

template<typename Array>
struct r2c_compute_indices_impl<Array, odd> {
    __device__ static void impl(Array& indices, int& rotation) {
        indices =
            detail::r2c_compute_offsets<Array::size, odd>();
        int warp_id = threadIdx.x & WARP_MASK;
        int size = Array::size;
        int r =
            size - detail::r2c_offset_constants<Array::size>::permute;
        rotation = (warp_id * r) % size;
    }
};

template<typename Array>
struct r2c_compute_indices_impl<Array, power_of_two> {
    __device__ static void impl(Array& indices, int& rotation) {
        int warp_id = threadIdx.x & WARP_MASK;
        int size = Array::size;
        rotation = warp_id % size;
        indices = r2c_compute_offsets_impl<Array, 0,
                                           Array::size, power_of_two>::impl(0);
    }
};

template<typename Array, typename Indices, typename Schema>
struct r2c_warp_transpose_impl {};

template<typename Array, typename Indices>
struct r2c_warp_transpose_impl<Array, Indices, odd> {
    __device__ static void impl(Array& src,
                                const Indices& indices,
                                const int& rotation) {
        Array rotated = rotate(src, rotation);
        detail::warp_shuffle<Array, Indices>::impl(rotated, indices);
        src = detail::r2c_tx_permute(rotated);
    }
};

template<typename Array, typename Indices>
struct r2c_warp_transpose_impl<Array, Indices, power_of_two> {
    __device__ static void impl(Array& src,
                                const Indices& indices,
                                const int& rotation) {
        Array rotated = rotate(src, rotation);
        detail::warp_shuffle<Array, Indices>::impl(rotated, indices);
        const int size = Array::size;
        int warp_id = threadIdx.x & WARP_MASK;
        src = rotate(detail::r2c_tx_permute(rotated),
                     (size-warp_id/(WARP_SIZE/size))%size);
    }
};

} //end namespace detail

template<typename Array>
__device__ void c2r_compute_indices(Array& indices, int& rotation) {
    detail::c2r_compute_indices_impl<
        Array,
        typename detail::tx_algorithm<Array::size>::type>
        ::impl(indices, rotation);
    
}

template<typename Array>
__device__ void c2r_warp_transpose(Array& src,
                                   const array<int, Array::size>& indices,
                                   int rotation) {    
    detail::c2r_warp_transpose_impl<
        Array, array<int, Array::size>,
        typename detail::tx_algorithm<Array::size>::type>::
        impl(src, indices, rotation);
}

template<typename Array>
__device__ void r2c_compute_indices(Array& indices, int& rotation) {
    detail::r2c_compute_indices_impl<
        Array, typename detail::tx_algorithm<Array::size>::type>
        ::impl(indices, rotation);

}

template<typename Array>
__device__ void r2c_warp_transpose(Array& src,
                                   const array<int, Array::size>& indices,
                                   int rotation) {
    detail::r2c_warp_transpose_impl<
        Array, array<int, Array::size>,
        typename detail::tx_algorithm<Array::size>::type>
    ::impl(src, indices, rotation);
}

} //end namespace trove
