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
"\n";
#endif
