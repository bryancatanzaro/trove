#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/generate.h>
#include <cstdlib>

#include <trove/transpose.h>
#include <trove/aos.h>
#include <trove/print_array.h>

#include <thrust/device_vector.h>



using namespace trove;

template<typename T>
__global__ void test_aos_gather(int* indices, T* s, T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    T data = load_aos(s, index);
    store_aos_warp_contiguous(data, r, global_index);
}

template<typename T>
__global__ void test_aos_scatter(int* indices, T* s, T* r) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    int index = indices[global_index];
    T data = load_aos_warp_contiguous(s, global_index);
    store_aos(data, r, index);
}

template<typename T>
void verify_gather(thrust::device_vector<int>& d_i,
                   thrust::device_vector<T>& d_s,
                   thrust::device_vector<T>& d_r) {
    thrust::host_vector<int> h_i = d_i;
    thrust::host_vector<T> h_s = d_s;
    thrust::host_vector<T> h_r = d_r;
    bool fail = false;
    for(int i = 0; i < h_r.size(); i++) {
        if (h_r[i] != h_s[h_i[i]]) {
            std::cout << "  Fail: r[" << i << "] is " << h_r[i] << std::endl;
            fail = true;
        }
    }
    if (!fail) {
        std::cout << "Pass!" << std::endl;
    }
}

template<typename T>
void verify_scatter(thrust::device_vector<int>& d_i,
                    thrust::device_vector<T>& d_s,
                    thrust::device_vector<T>& d_r) {
    thrust::host_vector<int> h_i = d_i;
    thrust::host_vector<T> h_s = d_s;
    thrust::host_vector<T> h_r = d_r;
    bool fail = false;
    for(int i = 0; i < h_r.size(); i++) {
        if (h_r[h_i[i]] != h_s[i]) {
            std::cout << "  Fail: r[" << h_i[i] << "] is " <<
                h_r[h_i[i]] << std::endl;
            fail = true;
        }
    }
    if (!fail) {
        std::cout << "Pass!" << std::endl;
    }
}

template<typename T, int i>
void print_warp_result(const thrust::device_vector<T> e) {
    thrust::host_vector<int> f = e;
    for(int row = 0; row < i; row++) {
        for(int col = 0; col < 32; col++) {
            int r = f[col + row * 32];
            if (r < 10) std::cout << " ";
            if (r < 100) std::cout << " ";
            std::cout << f[col + row * 32] << " ";
        }
        std::cout << std::endl;
    }
}

thrust::device_vector<int> make_device_random(int s) {
    thrust::host_vector<int> h(s);
    thrust::generate(h.begin(), h.end(), rand);
    thrust::device_vector<int> d = h;
    return d;
}

thrust::device_vector<int> make_random_permutation(int s) {
    thrust::device_vector<int> keys = make_device_random(s);
    thrust::counting_iterator<int> c(0);
    thrust::device_vector<int> values(s);
    thrust::copy(c, c+s, values.begin());
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    return values;
}

template<typename T>
void test_gather() {
    std::cout << "Testing gather" << std::endl;
    int n_blocks = 15 * 8 * 100;
    int block_size = 256;
    int n = n_blocks * block_size;
    int i = trove::detail::size_in_ints<T>::value;
    thrust::device_vector<int> d_s(n*i);
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end = begin + n*i;
    thrust::copy(begin, end, d_s.begin());
    thrust::device_ptr<T> s_begin((T*)thrust::raw_pointer_cast(d_s.data()));
    thrust::device_vector<T> s(n);
    thrust::copy(s_begin, s_begin + n, s.begin());
    thrust::device_vector<T> r(n);
    thrust::device_vector<int> indices = make_random_permutation(n);
    test_aos_gather
        <<<n_blocks, block_size>>>(thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(s.data()),
            thrust::raw_pointer_cast(r.data()));
    verify_gather(indices, s, r);
};

template<typename T>
void test_scatter() {
    std::cout << "Testing scatter" << std::endl;
    int n_blocks = 15 * 8 * 100;
    int block_size = 256;
    int n = n_blocks * block_size;
    int i = trove::detail::size_in_ints<T>::value;
    thrust::device_vector<int> d_s(n*i);
    thrust::counting_iterator<int> begin(0);
    thrust::counting_iterator<int> end = begin + n*i;
    thrust::copy(begin, end, d_s.begin());
    thrust::device_ptr<T> s_begin((T*)thrust::raw_pointer_cast(d_s.data()));
    thrust::device_vector<T> s(n);
    thrust::copy(s_begin, s_begin + n, s.begin());
    thrust::device_vector<T> r(n);
    thrust::device_vector<int> indices = make_random_permutation(n);
    test_aos_scatter
        <<<n_blocks, block_size>>>(thrust::raw_pointer_cast(indices.data()),
            thrust::raw_pointer_cast(s.data()),
            thrust::raw_pointer_cast(r.data()));
    verify_scatter(indices, s, r);
};



int main() {
    test_gather<array<int, 5> >();
    test_scatter<array<int, 5> >();
}

