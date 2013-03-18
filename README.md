Trove
=====

Trove is a CUDA library that provides efficient vector loads and
stores. It works for CUDA architectures 3.0 and above, and uses no
CUDA shared memory, making it easy to integrate.  It is useful for
working with data in an Array of Structures format, and also when
writing code that consumes or produces an array of data per CUDA
thread.


This functionality is built out of a transposition routine that uses
the warp shuffle intrinsic to redistribute data amongst threads in the
CUDA warp. For example, when every thread in
the warp is loading contiguous structures from an array, the threads
collaboratively load all the data needed by the warp, using coalesced
memory accesses, then transpose the data to redistribute it to the
correct thread.

The following cartoon illustrates how this works, for a notional warp
with eight threads.

![Transpose](https://raw.github.com/BryanCatanzaro/trove/master/doc/transpose.png)

Performance
===========

Accesses to arrays of structures can be 6X faster than direct memory
accesses using compiler generated loads and stores.

![Contiguous](https://raw.github.com/BryanCatanzaro/trove/master/doc/contiguous.png)
![Random](https://raw.github.com/BryanCatanzaro/trove/master/doc/random.png)

High-level Interface
====================

```c++
#include <trove/ptr.h>

template<typename T>
__global__ void
shfl_gather(const int length, const int* indices, T* raw_source, T* raw_dest) {
    int global_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (global_index < length) {
        int index = indices[global_index];
        trove::coalesced_ptr<T> source(raw_source);
        trove::coalesced_ptr<T> dest(raw_dest);
        T data = s[index];
        r[global_index] = data;
    }
}
```

The high-level interface allows you to load and store structures
directly. Just wrap the pointers of your arrays in
`trove::coalesced_ptr<T>`. You don't need to worry if the warp is
converged, or if the addresses you're accessing are contiguous. This
interface loses some performance, since it has to dynamically check
whether the warp is converged, and also broadcast all pointers from
all threads in each warp to all other threads in the warp, but it is
simple to use.

Low-level Interface
===================

If you know your warp is converged, and that your threads are all
accessing contiguous locations in your array, you can use the
low-level interface for maximum performance. By contiguous, we mean
that if threads with indices *i* and thread *j* are in the same warp,
the pointers *pi* and *pj* you pass to the library must obey the
relation *pj*-*pi* == *j* - *i*.  The low-level interface has the
following functions in `<trove/aos.h>`:

`template<typename T> __device__ T load_warp_contiguous(const T*
src);`

`template<typename T> __device__ void store_warp_contiguous(const T&
data, T* dest);`
