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

extern "C" __global__ void overlap_kernel(const int* __restrict__ E, const double* __restrict__ G, const int* __restrict__ N, double* __restrict__ O);
extern "C" __global__ void __launch_bounds__(128, 1) overlap_kernel(const int* __restrict__ E, const double* __restrict__ G, const int* __restrict__ N, double* __restrict__ O) {
  int count = 0;
  int next_count = 0;
  double area = 0x0p+0/*0.000000e+00*/;
  double ox = 0x0p+0/*0.000000e+00*/;
  double oy = 0x0p+0/*0.000000e+00*/;
  double ax = 0x0p+0/*0.000000e+00*/;
  double ay = 0x0p+0/*0.000000e+00*/;
  double bx = 0x0p+0/*0.000000e+00*/;
  double by = 0x0p+0/*0.000000e+00*/;
  double sx = 0x0p+0/*0.000000e+00*/;
  double sy = 0x0p+0/*0.000000e+00*/;
  double ex = 0x0p+0/*0.000000e+00*/;
  double ey = 0x0p+0/*0.000000e+00*/;
  double ds = 0x0p+0/*0.000000e+00*/;
  double de = 0x0p+0/*0.000000e+00*/;
  double ratio = 0x0p+0/*0.000000e+00*/;
  double height = 0x0p+0/*0.000000e+00*/;
  double volume = 0x0p+0/*0.000000e+00*/;
  double union_1 = 0x0p+0/*0.000000e+00*/;
  double px[16];
  double py[16];
  double qx[16];
  double qy[16];
  count = 4;
  next_count = 0;
  area = 0x0p+0/*0.000000e+00*/;
  ox = 0x0p+0/*0.000000e+00*/;
  oy = 0x0p+0/*0.000000e+00*/;
  ax = 0x0p+0/*0.000000e+00*/;
  ay = 0x0p+0/*0.000000e+00*/;
  bx = 0x0p+0/*0.000000e+00*/;
  by = 0x0p+0/*0.000000e+00*/;
  sx = 0x0p+0/*0.000000e+00*/;
  sy = 0x0p+0/*0.000000e+00*/;
  ex = 0x0p+0/*0.000000e+00*/;
  ey = 0x0p+0/*0.000000e+00*/;
  ds = 0x0p+0/*0.000000e+00*/;
  de = 0x0p+0/*0.000000e+00*/;
  ratio = 0x0p+0/*0.000000e+00*/;
  height = 0x0p+0/*0.000000e+00*/;
  volume = 0x0p+0/*0.000000e+00*/;
  union_1 = 0x0p+0/*0.000000e+00*/;
  O[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = 0x0p+0/*0.000000e+00*/;
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    ox = G[((((int)blockIdx.x) >> 1) * 12)];
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    oy = G[(((((int)blockIdx.x) >> 1) * 12) + 1)];
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    for (int k = 0; k < 4; ++k) {
      px[k] = (G[(((((int)blockIdx.x) >> 1) * 12) + (k * 2))] - ox);
      py[k] = (G[((((((int)blockIdx.x) >> 1) * 12) + (k * 2)) + 1)] - oy);
    }
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    for (int edge = 0; edge < 4; ++edge) {
      ax = (G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + (edge * 2))] - ox);
      ay = (G[(((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + (edge * 2)) + 1)] - oy);
      bx = (G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + (((edge + 1) & 3) * 2))] - ox);
      by = (G[(((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + (((edge + 1) & 3) * 2)) + 1)] - oy);
      next_count = 0;
      for (int k_1 = 0; k_1 < count; ++k_1) {
        int rmod = (((k_1 + count) - 1) % count);
        int rmod_1 = (((k_1 + count) - 1) % count);
        sx = px[((((0 <= count) && (0 <= rmod_1)) || ((count < 0) && (rmod_1 <= 0))) ? rmod_1 : (rmod_1 + count))];
        int rmod_2 = (((k_1 + count) - 1) % count);
        int rmod_3 = (((k_1 + count) - 1) % count);
        sy = py[((((0 <= count) && (0 <= rmod_3)) || ((count < 0) && (rmod_3 <= 0))) ? rmod_3 : (rmod_3 + count))];
        ex = px[k_1];
        ey = py[k_1];
        ds = (((bx - ax) * (sy - ay)) - ((by - ay) * (sx - ax)));
        de = (((bx - ax) * (ey - ay)) - ((by - ay) * (ex - ax)));
        if (((0x0p+0/*0.000000e+00*/ <= ds) && (de < 0x0p+0/*0.000000e+00*/)) || ((ds < 0x0p+0/*0.000000e+00*/) && (0x0p+0/*0.000000e+00*/ <= de))) {
          ratio = (ds / (ds - de));
        }
        if (((0x0p+0/*0.000000e+00*/ <= ds) && (de < 0x0p+0/*0.000000e+00*/)) || ((ds < 0x0p+0/*0.000000e+00*/) && (0x0p+0/*0.000000e+00*/ <= de))) {
          qx[next_count] = (sx + (ratio * (ex - sx)));
        }
        if (((0x0p+0/*0.000000e+00*/ <= ds) && (de < 0x0p+0/*0.000000e+00*/)) || ((ds < 0x0p+0/*0.000000e+00*/) && (0x0p+0/*0.000000e+00*/ <= de))) {
          qy[next_count] = (sy + (ratio * (ey - sy)));
        }
        if (((0x0p+0/*0.000000e+00*/ <= ds) && (de < 0x0p+0/*0.000000e+00*/)) || ((ds < 0x0p+0/*0.000000e+00*/) && (0x0p+0/*0.000000e+00*/ <= de))) {
          next_count = (next_count + 1);
        }
        if (0x0p+0/*0.000000e+00*/ <= de) {
          qx[next_count] = ex;
        }
        if (0x0p+0/*0.000000e+00*/ <= de) {
          qy[next_count] = ey;
        }
        if (0x0p+0/*0.000000e+00*/ <= de) {
          next_count = (next_count + 1);
        }
      }
      count = next_count;
      for (int k_2 = 0; k_2 < count; ++k_2) {
        px[k_2] = qx[k_2];
        py[k_2] = qy[k_2];
      }
    }
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    area = 0x0p+0/*0.000000e+00*/;
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    for (int k_3 = 0; k_3 < count; ++k_3) {
      int rmod_4 = ((k_3 + 1) % count);
      int rmod_5 = ((k_3 + 1) % count);
      int rmod_6 = ((k_3 + 1) % count);
      int rmod_7 = ((k_3 + 1) % count);
      area = ((area + (px[k_3] * py[((((0 <= count) && (0 <= rmod_5)) || ((count < 0) && (rmod_5 <= 0))) ? rmod_5 : (rmod_5 + count))])) - (py[k_3] * px[((((0 <= count) && (0 <= rmod_7)) || ((count < 0) && (rmod_7 <= 0))) ? rmod_7 : (rmod_7 + count))]));
    }
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    area = (fabs(area) * 0x1p-1/*5.000000e-01*/);
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    height = max(0x0p+0/*0.000000e+00*/, min(((G[(((((int)blockIdx.x) >> 1) * 12) + 8)] + (G[(((((int)blockIdx.x) >> 1) * 12) + 9)] * 0x1p-1/*5.000000e-01*/)) - (G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 8)] - (G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 9)] * 0x1p-1/*5.000000e-01*/))), ((G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 8)] + (G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 9)] * 0x1p-1/*5.000000e-01*/)) - (G[(((((int)blockIdx.x) >> 1) * 12) + 8)] - (G[(((((int)blockIdx.x) >> 1) * 12) + 9)] * 0x1p-1/*5.000000e-01*/)))));
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    volume = (area * height);
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    union_1 = ((((G[(((((int)blockIdx.x) >> 1) * 12) + 10)] * G[(((((int)blockIdx.x) >> 1) * 12) + 11)]) * G[(((((int)blockIdx.x) >> 1) * 12) + 9)]) + ((G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 10)] * G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 11)]) * G[((((((int)blockIdx.x) & 1) * 1536) + (((int)threadIdx.x) * 12)) + 9)])) - volume);
  }
  if ((((((int)blockIdx.x) >> 1) < N[0]) && ((((((int)blockIdx.x) & 1) * 128) + ((int)threadIdx.x)) < N[0])) && (E[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] != 0)) {
    O[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = (volume / (union_1 + 0x1.4f8b588e368f1p-17/*1.000000e-05*/));
  }
}

