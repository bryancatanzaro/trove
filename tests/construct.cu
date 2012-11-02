#include <trove/ptr.h>
#include "timer.h"

template<typename T,
         typename OutputIterator>
__global__ void
construct(OutputIterator b,
          int len) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int grid_size = gridDim.x * blockDim.x;
    
    for(int index = global_index; index < len; index += grid_size) {
        b[index] = T();
    }
}

#include <thrust/device_vector.h>


int main() {
    typedef double T;
    typedef trove::array<T, 3> n_array;
    int n = 100 * 8 * 256 * 15 - 100;
    thrust::device_vector<n_array> c(n);

    n_array* p_c(thrust::raw_pointer_cast(c.data()));
    trove::coalesced_ptr<n_array> c_c(p_c);

    int nthreads = 256;
    int nblocks = min(15 * 8, n/nthreads);
    
    int iterations = 10;

    std::cout << "Coalesced ";
    cuda_timer timer;
    timer.start();
    for(int j = 0; j < iterations; j++) {
        construct<n_array><<<nblocks, nthreads>>>(c_c, n);
    }
    float time = timer.stop();
    float gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    std::cout << "Direct ";
    timer.start();
    for(int j = 0; j < iterations; j++) {
        construct<n_array><<<nblocks, nthreads>>>(p_c, n);
    }
    time = timer.stop();
    gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
}

