#define lid(N) ((int) get_local_id(N))
#define gid(N) ((int) get_group_id(N))
#if __OPENCL_C_VERSION__ < 120
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

__kernel void __attribute__ ((reqd_work_group_size(1, 1, 1))) masssetupf(__global double *__restrict__ ctx, int const Q, __global int *__restrict__ iOf7, __global int *__restrict__ oOf7, __global double const *__restrict__ in, __global double *__restrict__ out)
{
  if (false)
  {
    ctx[0] = 0.0;
    iOf7[1] = 0;
    oOf7[0] = 0;
  }
  for (int i = 0; i <= -1 + Q; ++i)
  {
    out[oOf7[1] + i] = sqrt(in[i + iOf7[1]] * in[i + iOf7[1]] + in[Q + i + iOf7[1]] * in[Q + i + iOf7[1]] + in[2 * Q + i + iOf7[1]] * in[2 * Q + i + iOf7[1]]) * in[iOf7[2] + i] * (in[i + iOf7[1]] * (in[(3 + 1) * Q + i + iOf7[1]] * in[(2 * 3 + 2) * Q + i + iOf7[1]] + -1.0 * in[(3 + 2) * Q + i + iOf7[1]] * in[(2 * 3 + 1) * Q + i + iOf7[1]]) + -1.0 * in[Q + i + iOf7[1]] * (in[3 * Q + i + iOf7[1]] * in[(2 * 3 + 2) * Q + i + iOf7[1]] + -1.0 * in[(3 + 2) * Q + i + iOf7[1]] * in[Q * 2 * 3 + i + iOf7[1]]) + in[2 * Q + i + iOf7[1]] * (in[3 * Q + i + iOf7[1]] * in[(2 * 3 + 1) * Q + i + iOf7[1]] + -1.0 * in[(3 + 1) * Q + i + iOf7[1]] * in[Q * 2 * 3 + i + iOf7[1]]));
    out[oOf7[0] + i] = in[iOf7[2] + i] * (in[i + iOf7[1]] * (in[(3 + 1) * Q + i + iOf7[1]] * in[(2 * 3 + 2) * Q + i + iOf7[1]] + -1.0 * in[(3 + 2) * Q + i + iOf7[1]] * in[(2 * 3 + 1) * Q + i + iOf7[1]]) + -1.0 * in[Q + i + iOf7[1]] * (in[3 * Q + i + iOf7[1]] * in[(2 * 3 + 2) * Q + i + iOf7[1]] + -1.0 * in[(3 + 2) * Q + i + iOf7[1]] * in[Q * 2 * 3 + i + iOf7[1]]) + in[2 * Q + i + iOf7[1]] * (in[3 * Q + i + iOf7[1]] * in[(2 * 3 + 1) * Q + i + iOf7[1]] + -1.0 * in[(3 + 1) * Q + i + iOf7[1]] * in[Q * 2 * 3 + i + iOf7[1]]));
  }
}
