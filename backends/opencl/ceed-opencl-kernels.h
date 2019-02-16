#ifndef _ceed_opencl_kernels_h
#define _ceed_opencl_kernels_h

#include <stddef.h>

static const char *OpenCLKernels =
  "\n"
  "#define lid(N) ((int) get_local_id(N))\n"
  "#define gid(N) ((int) get_group_id(N))\n"
  "#if __OPENCL_C_VERSION__ < 120\n"
  "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
  "#endif\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict0(__global int const *__restrict__ indices, int const nelem_x_elemsize, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  for (int i = 0; i <= -1 + nelem_x_elemsize; ++i)\n"
  "    vv[i] = uu[indices[i]];\n"
  "}\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict2(int const elemsize, __global int const *__restrict__ indices, int const ncomp, int const nelem, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  for (int i = 0; i <= -1 + elemsize; ++i)\n"
  "    for (int e = 0; e <= -1 + nelem; ++e)\n"
  "      for (int d = 0; d <= -1 + ncomp; ++d)\n"
  "        vv[elemsize * ncomp * e + elemsize * d + i] = uu[ncomp * indices[elemsize * e + i] + d];\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict1(int const elemsize, __global int const *__restrict__ indices, int const ncomp, int const ndof, int const nelem, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  for (int i = 0; i <= -1 + elemsize; ++i)\n"
  "    for (int e = 0; e <= -1 + nelem; ++e)\n"
  "      for (int d = 0; d <= -1 + ncomp; ++d)\n"
  "        vv[elemsize * ncomp * e + elemsize * d + i] = uu[indices[elemsize * e + i] + ndof * d];\n"
  "}\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict3b(__global int const *__restrict__ indices, int const ndof, int const rng1, int const rngN, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  double acc_j;\n"
  "\n"
  "  for (int i = 0; i <= -1 + ndof; ++i)\n"
  "  {\n"
  "    acc_j = 0.0;\n"
  "    for (int j = rng1; j <= -1 + rngN; ++j)\n"
  "      acc_j = acc_j + uu[indices[j]];\n"
  "    vv[i] = acc_j;\n"
  "  }\n"
  "}\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict4b(int const elemsize, __global int const *__restrict__ indices, int const ncomp, int const ndof, int const rng1, int const rngN, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  double acc_j;\n"
  "\n"
  "  for (int i = 0; i <= -1 + ndof; ++i)\n"
  "    for (int d = 0; d <= -1 + ncomp; ++d)\n"
  "    {\n"
  "      acc_j = 0.0;\n"
  "      for (int j = rng1; j <= -1 + rngN; ++j)\n"
  "        acc_j = acc_j + uu[indices[j] * ncomp + d * elemsize + (indices[j] % elemsize)];\n"
  "      vv[ndof * d + i] = acc_j;\n"
  "    }\n"
  "}\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict5b(int const elemsize, __global int const *__restrict__ indices, int const ncomp, int const ndof, int const rng1, int const rngN, __global double const *__restrict__ uu, __global double *__restrict__ vv)\n"
  "{\n"
  "  double acc_j;\n"
  "\n"
  "  for (int i = 0; i <= -1 + ndof; ++i)\n"
  "    for (int d = 0; d <= -1 + ncomp; ++d)\n"
  "    {\n"
  "      acc_j = 0.0;\n"
  "      for (int j = rng1; j <= -1 + rngN; ++j)\n"
  "        acc_j = acc_j + uu[indices[j] * ncomp + d * elemsize + (indices[j] % elemsize)];\n"
  "      vv[ncomp * i + d] = acc_j;\n"
  "    }\n"
  "}\n"
  "\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kZero(int const elemsize, int const nc, int const nelem, __global double *__restrict__ v, int const vsize)\n"
  "{\n"
  "  for (int i = 0; i <= -1 + vsize; ++i)\n"
  "    for (int e = 0; e <= -1 + nelem; ++e)\n"
  "      v[e * nc * elemsize + i] = 0.0;\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kCeedTensorContract(int const A, char const Add, int const B, int const C, int const J, int const rxs_os, int const stride0, int const stride1, __global double const *__restrict__ t, char const transpose, __global double const *__restrict__ u, __global double *__restrict__ v, int const wxs_os)\n"
  "{\n"
  "  int rxs;\n"
  "  int tstride0;\n"
  "  int tstride1;\n"
  "  int wxs;\n"
  "\n"
  "  tstride1 = (transpose ? J : 1);\n"
  "  tstride0 = (transpose ? 1 : B);\n"
  "  for (int j = 0; j <= -1 + J; ++j)\n"
  "    for (int c = 0; c <= -1 + C; ++c)\n"
  "      for (int a = 0; a <= -1 + A; ++a)\n"
  "      {\n"
  "        wxs = (a * J + j) * C + c + wxs_os;\n"
  "        for (int b = 0; b <= -1 + B; ++b)\n"
  "        {\n"
  "          rxs = (a * B + b) * C + c + rxs_os;\n"
  "          if (!Add)\n"
  "            v[wxs] = t[j * stride0 + b * stride1] * u[rxs];\n"
  "          if (Add)\n"
  "            v[wxs] = v[wxs] + t[j * stride0 + b * stride1] * u[rxs];\n"
  "        }\n"
  "      }\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kInterp(int const P1d, int const Q1d, int const QnD, __global double const *__restrict__ d_u, __global double *__restrict__ d_v, int const dim, int const elemsize, __global double const *__restrict__ interp1d, int const nc, int const ndof, int const nelem, int const nqpt, int const stride0, int const stride1, __global double *__restrict__ tmp0, __global double *__restrict__ tmp1, char const transpose)\n"
  "{\n"
  "  int P;\n"
  "  int Q;\n"
  "  int d_u_offset;\n"
  "  int d_v_offset;\n"
  "  int indr;\n"
  "  int indw;\n"
  "  int post;\n"
  "  int pre;\n"
  "  int rxs;\n"
  "  int tstride0;\n"
  "  int tstride1;\n"
  "  int u_offset;\n"
  "  int v_offset;\n"
  "  int wxs;\n"
  "\n"
  "  Q = (transpose ? P1d : Q1d);\n"
  "  P = (transpose ? Q1d : P1d);\n"
  "  for (int e = 0; e <= -1 + nelem; ++e)\n"
  "  {\n"
  "    v_offset = e * QnD * nc * (dim + 2);\n"
  "    u_offset = e * nc * elemsize;\n"
  "    d_v_offset = (transpose ? u_offset : v_offset);\n"
  "    d_u_offset = (transpose ? v_offset : u_offset);\n"
  "    for (int d = 0; d <= -1 + dim; ++d)\n"
  "    {\n"
  "      post = (int)pow((double)Q, (double)d);\n"
  "      pre = ndof * (int)pow((double)P, (double)(dim + -1 + -1 * d));\n"
  "      for (int j = 0; j <= -1 + Q; ++j)\n"
  "        for (int c = 0; c <= -1 + post; ++c)\n"
  "          for (int b = 0; b <= -1 + P; ++b)\n"
  "            for (int a = 0; a <= -1 + pre; ++a)\n"
  "            {\n"
  "              tstride1 = (transpose ? Q : 1);\n"
  "              tstride0 = (transpose ? 1 : P);\n"
  "              indr = (a * P + b) * post + c;\n"
  "              rxs = indr + d_v_offset;\n"
  "              indw = (a * Q + j) * post + c;\n"
  "              if (d == 0 && !(d == dim + -1))\n"
  "                tmp1[indw] = interp1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "              wxs = indw + d_u_offset;\n"
  "              if (d == dim + -1)\n"
  "              {\n"
  "                if ((d % 2) == 0 && !(d == 0))\n"
  "                  d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                if (d == 0)\n"
  "                  d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "              }\n"
  "              if (!(d == 0))\n"
  "              {\n"
  "                if (!(d == dim + -1))\n"
  "                {\n"
  "                  if ((d % 2) == 0)\n"
  "                    tmp0[indw] = interp1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                  if (!((d % 2) == 0))\n"
  "                    tmp1[indw] = interp1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "                }\n"
  "                if (!((d % 2) == 0) && d == dim + -1)\n"
  "                  d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "              }\n"
  "            }\n"
  "    }\n"
  "    if (!transpose)\n"
  "      d_v_offset = d_v_offset + nqpt;\n"
  "    if (transpose)\n"
  "      d_u_offset = d_u_offset + nqpt;\n"
  "  }\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kGrad(int const P1d, int const Q1d, int const QnD, __global double const *__restrict__ d_u, __global double *__restrict__ d_v, int const dim, int const elemsize, __global double const *__restrict__ grad1d, __global double const *__restrict__ interp1d, int const nc, int const ndof, int const nelem, int const nqpt, int const stride0, int const stride1, __global double *__restrict__ tmp0, __global double *__restrict__ tmp1, char const transpose)\n"
  "{\n"
  "  int P;\n"
  "  int Q;\n"
  "  int d_u_offset;\n"
  "  int d_v_offset;\n"
  "  int indr;\n"
  "  int indw;\n"
  "  int post;\n"
  "  int pre;\n"
  "  int rxs;\n"
  "  int tstride0;\n"
  "  int tstride1;\n"
  "  int u_offset;\n"
  "  int v_offset;\n"
  "  int wxs;\n"
  "\n"
  "  Q = (transpose ? P1d : Q1d);\n"
  "  P = (transpose ? Q1d : P1d);\n"
  "  for (int e = 0; e <= -1 + nelem; ++e)\n"
  "  {\n"
  "    v_offset = e * QnD * nc * (dim + 2);\n"
  "    u_offset = e * nc * elemsize;\n"
  "    d_v_offset = (transpose ? u_offset : v_offset);\n"
  "    d_u_offset = (transpose ? v_offset : u_offset);\n"
  "    for (int p = 0; p <= -1 + dim; ++p)\n"
  "      for (int d = 0; d <= -1 + dim; ++d)\n"
  "      {\n"
  "        post = (int)pow((double)Q, (double)d);\n"
  "        pre = ndof * (int)pow((double)P, (double)(dim + -1 + -1 * d));\n"
  "        for (int j = 0; j <= -1 + Q; ++j)\n"
  "          for (int c = 0; c <= -1 + post; ++c)\n"
  "            for (int b = 0; b <= -1 + P; ++b)\n"
  "              for (int a = 0; a <= -1 + pre; ++a)\n"
  "              {\n"
  "                tstride1 = (transpose ? Q : 1);\n"
  "                tstride0 = (transpose ? 1 : P);\n"
  "                indr = (a * P + b) * post + c;\n"
  "                rxs = indr + d_v_offset;\n"
  "                indw = (a * Q + j) * post + c;\n"
  "                if (p == d && d == 0 && !(d == dim + -1))\n"
  "                  tmp1[indw] = grad1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "                wxs = indw + d_u_offset;\n"
  "                if (p == d && d == dim + -1)\n"
  "                {\n"
  "                  if ((d % 2) == 0 && !(d == 0))\n"
  "                    d_v[wxs] = transpose * d_v[wxs] + grad1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                  if (d == 0)\n"
  "                    d_v[wxs] = transpose * d_v[wxs] + grad1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "                }\n"
  "                if (!(p == d))\n"
  "                {\n"
  "                  if (!(d == 0))\n"
  "                  {\n"
  "                    if (!(d == dim + -1))\n"
  "                    {\n"
  "                      if (!((d % 2) == 0))\n"
  "                        tmp1[indw] = interp1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "                      if ((d % 2) == 0)\n"
  "                        tmp0[indw] = interp1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                    }\n"
  "                    if (d == dim + -1)\n"
  "                    {\n"
  "                      if (!((d % 2) == 0))\n"
  "                        d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "                      if ((d % 2) == 0)\n"
  "                        d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                    }\n"
  "                  }\n"
  "                  if (d == 0)\n"
  "                  {\n"
  "                    if (!(d == dim + -1))\n"
  "                      tmp1[indw] = interp1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "                    if (d == dim + -1)\n"
  "                      d_v[wxs] = transpose * d_v[wxs] + interp1d[j * stride0 + b * stride1] * d_u[rxs];\n"
  "                  }\n"
  "                }\n"
  "                if (p == d && !(d == 0))\n"
  "                {\n"
  "                  if (!(d == dim + -1))\n"
  "                  {\n"
  "                    if (!((d % 2) == 0))\n"
  "                      tmp1[indw] = grad1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "                    if ((d % 2) == 0)\n"
  "                      tmp0[indw] = grad1d[j * stride0 + b * stride1] * tmp1[indr];\n"
  "                  }\n"
  "                  if (!((d % 2) == 0) && d == dim + -1)\n"
  "                    d_v[wxs] = transpose * d_v[wxs] + grad1d[j * stride0 + b * stride1] * tmp0[indr];\n"
  "                }\n"
  "              }\n"
  "      }\n"
  "    if (!transpose)\n"
  "      d_v_offset = d_v_offset + nqpt;\n"
  "    if (transpose)\n"
  "      d_u_offset = d_u_offset + nqpt;\n"
  "  }\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kWeight(int const Q, int const QnD, __global double *__restrict__ d_v, int const dim, int const nc, int const nelem, __global double const *__restrict__ qweight1d)\n"
  "{\n"
  "  int v_offset;\n"
  "  int v_shift;\n"
  "  int xs;\n"
  "\n"
  "  v_shift = QnD * nc + QnD * nc * dim;\n"
  "  for (int k = 0; k <= -1 + Q; ++k)\n"
  "    for (int j = 0; j <= -1 + Q; ++j)\n"
  "      for (int i = 0; i <= -1 + Q; ++i)\n"
  "        for (int e = 0; e <= -1 + nelem; ++e)\n"
  "        {\n"
  "          v_offset = e * QnD * nc * (dim + 2) + v_shift;\n"
  "          xs = (i * Q + j) * Q + k + v_offset;\n"
  "          for (int d = 0; d <= -1 + dim; ++d)\n"
  "          {\n"
  "            if (!(d == 0))\n"
  "              d_v[xs] = qweight1d[j] * d_v[xs];\n"
  "            if (d == 0)\n"
  "              d_v[xs] = qweight1d[j];\n"
  "          }\n"
  "        }\n"
  "}\n"
  "__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) setup(__global int const *__restrict__ ctx, int const Q, __global int const *__restrict__ iOf7, __global int const *__restrict__ oOf7, __global double const *__restrict__ in, __global double *__restrict__ out)\n"
  "{\n"
  " for (int i = 0; i <= -1 + Q; ++i)\n"
  "      out[i + oOf7[0]] = in[i + iOf7[0]];\n"
  "}\n"
"__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) mass(__global int const *__restrict__ ctx, int const Q, __global int const *__restrict__ iOf7, __global int const *__restrict__ oOf7, __global double const *__restrict__ in, __global double *__restrict__ out)\n"
"{\n"
"      for (int i = 0; i <= -1 + Q; ++i)\n"
"           out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]];\n"
"}\n"
"\n";
#endif
