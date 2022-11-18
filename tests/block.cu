#include <iostream>
#include <trove/block.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>

// tile size to use for low-level load/store warp contiguous
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif

template<typename T, int s>
__global__ void test_block_write(T* r, int l) {
    typedef trove::array<T, s> s_ary;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    for(int index = global_index; index < l; index += gridDim.x * blockDim.x) {
        //Generate some test data to write out
        s_ary d = trove::counting_array<s_ary>::impl(s * index);

        //The high performance vector memory accesses only function correctly
        //if the warp is converged. Here we check.
        if (trove::warp_converged()) {
            //Warp converged, indices are contiguous, so we call the
            //fast store            
            trove::store_array_warp_contiguous<s, TILE_SIZE>(r, index, d);
        } else {
            //Warp is not converged, call the slow store.
            trove::store_array(r, index, d);
        }
    }
}

template<typename T, int s>
__global__ void test_block_copy(const T* x, T* r, int l) {
    typedef trove::array<T, s> s_ary;
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;

    for(int index = global_index; index < l; index += gridDim.x * blockDim.x) {

        //The high performance vector memory accesses only function
        //correctly if the warp is converged. Here we check.
        if (trove::warp_converged()) {
            //Warp converged, indices are contiguous, call the fast
            //load and store
            s_ary d = trove::load_array_warp_contiguous<s, TILE_SIZE>(x, index);
            trove::store_array_warp_contiguous(r, index, d);
        } else {
            //Warp not converged, call the slow load and store
            s_ary d = trove::load_array<s>(x, index);
            trove::store_array(r, index, d);
        }
    }
}

int main() {
    int l = 100000 * 256 + 17;
    int int_length = l * 5;
    thrust::device_vector<int> d(int_length);
    test_block_write<int, 5><<<100*15, 256>>>(
        thrust::raw_pointer_cast(d.data()), l);
    thrust::device_vector<int> g(int_length);
    thrust::counting_iterator<int> c(0);
    thrust::copy(c, c + int_length, g.begin()); 
    std::cout << "test_block_write results pass: " << std::boolalpha <<
        thrust::equal(d.begin(), d.end(), g.begin()) << std::endl;
    thrust::device_vector<int> e(int_length);
    test_block_copy<int, 5><<<100*15, 256>>>(
        thrust::raw_pointer_cast(g.data()),
        thrust::raw_pointer_cast(e.data()),
        l);
    std::cout << "test_block_copy results pass: " << std::boolalpha <<
        thrust::equal(e.begin(), e.end(), g.begin()) << std::endl;
    
}
