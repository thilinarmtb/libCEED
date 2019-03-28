#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) kRestrict6(__global double const *__restrict__ uu, __global double *__restrict__ vv)
{
  for (int i = 0; i <= 23; ++i)
    vv[i] = uu[i];
}

