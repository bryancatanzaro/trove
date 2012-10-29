#include <trove/ptr.h>


template<typename T,
         typename OutputIterator>
__global__ void
//__launch_bounds__(256, 8)
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
    typedef trove::array<double, 24> n_array;
    int n = 100 * 8 * 256 * 15;
    thrust::device_vector<n_array> c(n);

    trove::coalesced_ptr<n_array> c_c(thrust::raw_pointer_cast(c.data()));

    int nthreads = 256;
    int nblocks = min(15 * 8, n/nthreads);
    
    int iterations = 10;
    cudaEvent_t start, stop;
    float time = 0;
    std::cout << "Coalesced ";
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int j = 0; j < iterations; j++) {
        construct<n_array><<<nblocks, nthreads>>>(c_c, n);
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
        construct<n_array><<<nblocks, nthreads>>>(p_c, n);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    gbs = (float)(sizeof(n_array) * (iterations * (n + 1))) / (time * 1000000);
    std::cout << gbs << std::endl;

    
}

