#include <stdio.h>
#include <trove/ptr.h>

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
struct euclidean_distance {
    static const int i = A::size;
    typedef typename A::value_type T;
    typedef A input_type;
    typedef T result_type;
    
    input_type m_point;
    __host__ __device__ euclidean_distance(const input_type& point) : m_point(point) {}

    template<int j>
    __device__
    static T diff2(const T& t, const T& p) {
        T diff = t - p;
        return diff * diff;
    }
    
    __device__
    static T impl(const trove::array<T, 1>& a, const trove::array<T, 1>& p) {
        return diff2<i-1>(a.head, p.head);
    }

    template<int j>
    __device__
    static T impl(const trove::array<T, j>& a, const trove::array<T, j>& p) {
        return diff2<i-j>(a.head, p.head) + impl(a.tail, p.tail);
    }

    __device__ T operator()(const input_type& o) {
        return impl(o, m_point);
    }
};

template<
    typename Fn,
    typename InputIterator,
    typename OutputIterator>
__global__ void
//__launch_bounds__(256, 8)
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
    typedef trove::array<T, 3> n_array;
    int n = 100 * 8 * 256 * 15;
    thrust::device_vector<n_array> c = make_filled<n_array>(n);
    trove::coalesced_ptr<n_array> c_c(thrust::raw_pointer_cast(c.data()));
    thrust::device_vector<T> r(n);
    T* d_r = thrust::raw_pointer_cast(r.data());

    n_array center = trove::counting_array<n_array >::impl();
    
    euclidean_distance<n_array> fn(center);
    
    int nthreads = 256;
    int nblocks = min(15 * 8, n/nthreads);
    
    int iterations = 1;
    cudaEvent_t start, stop;
    float time = 0;
    std::cout << "Coalesced ";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int j = 0; j < iterations; j++) {
        transform<<<nblocks, nthreads>>>(fn, c_c, d_r, n);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    float gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
    n_array* p_c(thrust::raw_pointer_cast(c.data()));

    std::cout << "Direct ";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int j = 0; j < iterations; j++) {
        transform<<<nblocks, nthreads>>>(fn, p_c, d_r, n);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
}

