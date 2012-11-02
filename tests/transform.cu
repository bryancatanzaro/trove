#include <iostream>
#include <trove/ptr.h>
#include "timer.h"

template<typename P>
struct value_type{};

template<typename P>
struct value_type<P*> {
    typedef P type;
};

template<typename P>
struct value_type<trove::coalesced_ptr<P> > {
    typedef P type;
};

template<typename A>
struct update {
    typedef typename A::value_type T;
    typedef A result_type;
    
    T m_value;
    __host__ __device__ update(const T& value) : m_value(value) {}

    __device__
    static trove::array<T, 1> impl(const trove::array<T, 1>& a, const T& value) {
        return trove::array<T, 1>(trove::get<0>(a) + value);
    }

    template<int j>
    __device__
    static trove::array<T, j> impl(const trove::array<T, j>& a, const T& value) {
        return trove::array<T, j>(trove::get<0>(a) + value,
                                  impl(a.tail, value));
    }

    __device__ A operator()(const A& o) {
        return impl(o, m_value);
    }
};

template<
    typename Fn,
    typename InputIterator,
    typename OutputIterator>
__global__ void
    transform(Fn f,
              InputIterator input,
              OutputIterator output,
              int len) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    
    for(int index = global_index; index < len; index += grid_size) {
        output[index] = f(input[index]);
    }
}

#include <thrust/device_vector.h>

template<typename A>
thrust::device_vector<A> make_filled(int n) {
    typedef typename A::value_type T;
    thrust::device_vector<A> result(n);
    thrust::device_ptr<T> p = thrust::device_ptr<T>((T*)thrust::raw_pointer_cast(result.data()));
    thrust::counting_iterator<T> c(0);
    int s = n * A::size;
    thrust::copy(c, c+s, p);
    return result;
}

int main() {
    typedef double T;
    typedef trove::array<T, 6> n_array;
    int n = 100 * 8 * 256 * 15;
    thrust::device_vector<n_array> c = make_filled<n_array>(n);
    n_array* d_c = thrust::raw_pointer_cast(c.data());
    trove::coalesced_ptr<n_array> c_c(d_c);

    thrust::device_vector<n_array> r(n);
    n_array* d_r = thrust::raw_pointer_cast(r.data());
    trove::coalesced_ptr<n_array> c_r(d_r);
    
    T value = 1;

    update<n_array> fn(value);
    
    int nthreads = 256;
    int nblocks = min(15 * 8, n/nthreads);
    
    int iterations = 1;
    std::cout << "Coalesced ";
    cuda_timer timer;
    timer.start();
    for(int j = 0; j < iterations; j++) {
        transform<<<nblocks, nthreads>>>(fn, c_c, c_r, n);
    }
    float time = timer.stop();
    float gbs = (float)(sizeof(n_array) * (iterations * (2*n))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
    std::cout << "Direct ";
    timer.start();
    for(int j = 0; j < iterations; j++) {
        transform<<<nblocks, nthreads>>>(fn, d_c, d_r, n);
    }
    time = timer.stop();
    gbs = (float)(sizeof(n_array) * (iterations * (2*n))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
}

