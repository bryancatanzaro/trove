#include <iostream>
#include <trove/block.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>

template<typename T, int s>
__global__ void test_block_write(T* r) {
    typedef trove::array<T, s> s_ary;
    s_ary d = trove::counting_array<s_ary>::impl(1);
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    trove::store_array_warp_contiguous(r, global_index, d);
}

template<typename T, int s>
struct repeater {
    typedef T result_type;
    __host__ __device__ result_type operator()(int i) {
        return i % s;
    }
};

int main() {
    int l = 100 * 256 * 5;
    thrust::device_vector<int> d(l);
    test_block_write<int, 5><<<100, 256>>>(
        thrust::raw_pointer_cast(d.data()));
    thrust::device_vector<int> g(l);
    thrust::counting_iterator<int> c(0);
    thrust::transform(c, c + l, g.begin(), repeater<int, 5>()); 
    std::cout << "Check: " << std::boolalpha <<
        thrust::equal(d.begin(), d.end(), g.begin()) << std::endl;
    
    
}
