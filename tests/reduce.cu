#include <trove/ptr.h>

template<typename T>
struct array_sum {
    typedef T result_type;

    template<typename E>
    __device__
    static trove::array<E, 1> impl(const trove::array<E, 1>& a, const trove::array<E, 1>& b) {
        return trove::array<E, 1>(a.head + b.head);
    }

    template<typename E, int i>
    __device__
    static trove::array<E, i> impl(const trove::array<E, i>& a, const trove::array<E, i>& b) {
        return trove::array<E, i>(a.head + b.head,
                                  impl(a.tail, b.tail));
    }
    
    __device__
    T operator()(const T& a, const T& b) {
        return impl(a, b);
    }
};


template<int i, typename Fn, typename T>
struct warp_reduce_step {
    __device__ static void impl(int warp_id, Fn fn, T& accumulator) {
        int other_idx = warp_id + i/2;
        T other = __shfl(accumulator, other_idx);
        if (warp_id < i/2) accumulator = fn(accumulator, other);
        warp_reduce_step<i/2, Fn, T>::impl(warp_id, fn, accumulator);
    }
};

template<typename Fn, typename T>
struct warp_reduce_step<0, Fn, T> {
    __device__ static void impl(int, Fn, T&) {}
};



template<typename Fn, typename T>
__device__ void warp_reduce(Fn fn, T& accumulator) {
    int warp_id = threadIdx.x & WARP_MASK;
    warp_reduce_step<WARP_SIZE/2, Fn, T>::impl(warp_id, fn, accumulator);
}

template<
    int block_size,
    typename InputIterator,
    typename OutputIterator,
    typename Fn>
__global__ void
__launch_bounds__(block_size, 2048/block_size)
reduce(InputIterator a,
       OutputIterator b,
       int len,
       Fn fn) {
    //__shared__ Fn::result_type scratch[block_size / WARP_SIZE];
    typedef typename Fn::result_type T;
    T accumulator;
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    
    for(int index = global_index; index < len; index += grid_size) {
        accumulator = fn(accumulator, a[index]);
    }
    warp_reduce(fn, accumulator);
    int warp_idx = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
    b[warp_idx] = accumulator;
}

#include <thrust/device_vector.h>

template<typename T>
thrust::device_vector<T> make_filled(int n) {
    thrust::device_vector<T> result(n);
    thrust::device_ptr<int> p = thrust::device_ptr<int>((int*)thrust::raw_pointer_cast(result.data()));
    thrust::counting_iterator<int> c(0);
    int s = n * sizeof(T) / sizeof(int);
    thrust::copy(c, c+s, p);
    return result;
}

int main() {
    typedef trove::array<int, 7> n_array;
    int n = 100 * 8 * 256 * 15;
    thrust::device_vector<n_array> a = make_filled<n_array>(n);
    thrust::device_vector<n_array> c(n/WARP_SIZE);

    trove::coalesced_ptr<const n_array> c_a(thrust::raw_pointer_cast(a.data()));
    trove::coalesced_ptr<n_array> c_c(thrust::raw_pointer_cast(c.data()));

    array_sum<n_array> fn;


    int iterations = 10;
    cudaEvent_t start, stop;
    float time = 0;
    std::cout << "Coalesced ";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int j = 0; j < iterations; j++) {
        reduce<256><<<((n-1)/256)+1, 256>>>(c_a, c_c, n, fn);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    float gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
    const n_array* p_a(thrust::raw_pointer_cast(a.data()));
    n_array* p_c(thrust::raw_pointer_cast(c.data()));

    std::cout << "Direct ";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int j = 0; j < iterations; j++) {
        reduce<256><<<((n-1)/256)+1, 256>>>(p_a, p_c, n, fn);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
}

