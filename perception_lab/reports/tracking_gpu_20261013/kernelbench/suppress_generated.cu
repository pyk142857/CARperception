#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void suppress_kernel(const double* __restrict__ I, int64_t* __restrict__ Keep, const int* __restrict__ N, const int64_t* __restrict__ Order);
extern "C" __global__ void __launch_bounds__(32, 1) suppress_kernel(const double* __restrict__ I, int64_t* __restrict__ Keep, const int* __restrict__ N, const int64_t* __restrict__ Order) {
  int kept = 0;
  int chosen = 0;
  int removed[256];
  kept = 0;
  chosen = 0;
  if (((int)threadIdx.x) == 0) {
    for (int k = 0; k < 256; ++k) {
      removed[k] = 0;
      Keep[k] = (int64_t)-1;
    }
    for (int position = 0; position < N[0]; ++position) {
      int64_t condval;
      if ((position < 256)) {
        condval = Order[position];
      } else {
        condval = (int64_t)0;
      }
      chosen = ((int)condval);
      bool if_condition = (removed[chosen] == 0);
      if (if_condition) {
        if (0 <= kept) {
          if (kept < 257) {
            Keep[kept] = ((int64_t)chosen);
          }
        }
        kept = (kept + 1);
        for (int candidate = 0; candidate < N[0]; ++candidate) {
          double condval_1;
          if ((((candidate < 256) && (0 <= chosen)) && (chosen < 256))) {
            condval_1 = I[((((int64_t)chosen) * (int64_t)256) + ((int64_t)candidate))];
          } else {
            condval_1 = 0x0p+0/*0.000000e+00*/;
          }
          if (0x1.999999999999ap-4/*1.000000e-01*/ < condval_1) {
            removed[candidate] = 1;
          }
        }
      }
    }
    Keep[256] = ((int64_t)kept);
  }
}

