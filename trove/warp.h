/*
Copyright (c) 2013, NVIDIA Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <cooperative_groups.h>

namespace trove {

enum {
  WARP_SIZE = 32,
  WARP_CONVERGED = 0xFFFFFFFF
};

__device__ constexpr size_t log2(size_t n) { return ( n < 2 ? 0 : 1 + log2(n / 2)); }

namespace cg = cooperative_groups;

template <size_t threads>
struct thread_tile {
  cg::thread_block_tile<threads> tile;
  __device__ thread_tile() : tile(cg::tiled_partition<threads>(cg::this_thread_block())) { }

  template <typename T> __device__ T shfl(const T& t, const int& i) const { return tile.shfl(t, i); }

  __device__ static constexpr auto size() { return decltype(tile)::num_threads(); }
  __device__ static constexpr auto log_size() { return log2(size()); }
  __device__ static constexpr auto mask() { return size() - 1; }
  __device__ auto id() const { return tile.thread_rank(); }
};

__device__ inline bool warp_converged() { return (__activemask() == WARP_CONVERGED); }

__device__ inline bool half_warp_converged()
{
  auto lane_id = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & 31;
  auto shift = lane_id & ~0xe;
  auto lane_mask = 65535 << shift;
  return (__activemask() & lane_mask) == lane_mask;
}

__device__ inline bool quarter_warp_converged()
{
  auto lane_id = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & 31;
  auto shift = lane_id & ~0x7;
  auto lane_mask = 255 << shift;
  return (__activemask() & lane_mask) == lane_mask;
}

__device__ inline bool eighth_warp_converged()
{
  auto lane_id = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) & 31;
  auto shift = lane_id & ~0x3;
  auto lane_mask = 15 << shift;
  return (__activemask() & lane_mask) == lane_mask;
}

}
