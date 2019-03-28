#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict2(__global int const *__restrict__ indices, __global double const *__restrict__ uu, __global double *__restrict__ vv)
{
  for (int i = 0; i <= 7; ++i)
  {
    int const e = 0;

    for (int d = 0; d <= 2; ++d)
      vv[8 * d + i] = uu[3 * indices[i] + d];
  }
}

