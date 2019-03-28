#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) massf(__global double *__restrict__ ctx, int const Q, __global int const *__restrict__ iOf7, __global int const *__restrict__ oOf7, __global double const *__restrict__ in, __global double *__restrict__ out)
{
  if (false)
    ctx[0] = 0.0;
  for (int i = 0; i <= -1 + Q; ++i)
    out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]];
}
