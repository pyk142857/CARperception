/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// Single candidate: precompute sorted feature offsets once per forward.
// Native NHWC FP16 features / NDHW FP16 depth / NCHW FP16 output.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

static __global__ void preindex_kernel(const unsigned int* indices,
    unsigned int* offsets, unsigned int count, unsigned int depth,
    unsigned int area, unsigned int channels) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    unsigned int rank = indices[i];
    offsets[i] = (rank / (depth * area) * area + rank % area) * channels;
  }
}
struct __align__(4) Half10 { half val[10]; };
static __global__ void preindexed_pool_kernel(const half* features,
    const half* depth_weights, const unsigned int* indices,
    const unsigned int* offsets, const int3* intervals, unsigned int interval_count,
    unsigned int output_area, half* output) {
  unsigned int interval_index = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int feature_block = threadIdx.x * 10;
  if (interval_index >= interval_count) return;
  int3 interval = intervals[interval_index];
  float accumulate[10] = {0.0f};
  for (int i = interval.x; i < interval.y; ++i) {
    unsigned int rank = indices[i];
    float weight = __half2float(depth_weights[rank]);
    Half10 feature = *reinterpret_cast<const Half10*>(features + offsets[i] + feature_block);
#pragma unroll
    for (int j = 0; j < 10; ++j)
      accumulate[j] = fma(__half2float(feature.val[j]), weight, accumulate[j]);
  }
#pragma unroll
  for (int j = 0; j < 10; ++j)
    output[interval.z + (feature_block + j) * output_area] = __float2half(accumulate[j]);
}

// Full candidate includes offset kernel, output memset and pooling kernel.
// Workspace is caller-owned and persistent; no geometry is reused across calls.
extern "C" int car_bev_pool_preindex_forward(
    const void* features, const void* depth_weights, const void* indices,
    const void* intervals, unsigned int interval_count, unsigned int C,
    unsigned int D, unsigned int H, unsigned int W, unsigned int out_h,
    unsigned int out_w, unsigned int index_count, void* offsets, void* output,
    void* stream) {
  if (C != 80 || !D || !H || !W || !out_h || !out_w)
    return static_cast<int>(cudaErrorInvalidValue);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  cudaError_t error = cudaMemsetAsync(output, 0, C*out_h*out_w*sizeof(half), s);
  if (error != cudaSuccess || !interval_count) return static_cast<int>(error);
  if (!index_count) return static_cast<int>(cudaErrorInvalidValue);
  preindex_kernel<<<(index_count+255)/256,256,0,s>>>(
      static_cast<const unsigned int*>(indices),static_cast<unsigned int*>(offsets),
      index_count,D,H*W,C);
  error = cudaGetLastError();
  if (error != cudaSuccess) return static_cast<int>(error);
  // Exactly original pack10 launch dimensions: 8 x 128 = 1024 threads/block.
  dim3 threads(C/10,1024/(C/10));
  dim3 blocks(1,(interval_count+threads.y-1)/threads.y);
  preindexed_pool_kernel<<<blocks,threads,0,s>>>(
      static_cast<const half*>(features),static_cast<const half*>(depth_weights),
      static_cast<const unsigned int*>(indices),static_cast<const unsigned int*>(offsets),
      static_cast<const int3*>(intervals),interval_count,out_h*out_w,static_cast<half*>(output));
  return static_cast<int>(cudaGetLastError());
}
