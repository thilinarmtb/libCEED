#include <stdio.h>
#include <math.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
    "__kernel void vecAdd(  __global double *a,                       \n" \
    "                       __global double *b,                       \n" \
    "                       __global double *c,                       \n" \
    "                       const unsigned int n)                    \n" \
    "{                                                               \n" \
    "    //Get our global thread ID                                  \n" \
    "    int id = get_global_id(0);                                  \n" \
    "    //printf(\"%d\", 99999)                                     \n" \
    "    //Make sure we do not go out of bounds                      \n" \
    "    if (id < n)                                                 \n" \
    "        c[id] = a[id] + b[id];                                  \n" \
    "}                                                               \n" \
    "\n" ;

int OpenCL_test_00(void) {
  // Length of vectors
  
  unsigned int n = 100000;

  // Host input vectors
  double *h_a;
  double *h_b;
  // Host output vector
  double *h_c;

  // Device input buffers
  cl_mem d_a;
  cl_mem d_b;
  // Device output buffer
  cl_mem d_c;

  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double);

  // Allocate memory for each vector on host
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);

  // Initialize vectors on host
  unsigned int i;
  for( i = 0; i < n; i++ ) {
    h_a[i] = sin(i)*sin(i);
    h_b[i] = cos(i)*cos(i);
  }

  size_t globalSize, localSize;
  cl_int err;

  // Number of work items in each local work group
  localSize = 64;

  // Number of total work items - localSize must be devisor
  globalSize = ceil(n/(double)localSize)*localSize;
  
  // Get platform and device information
  cl_platform_id* platforms = NULL;
  cl_uint num_platforms;

  //Set up the Platform
  cl_int clStatus = clGetPlatformIDs(0, NULL, &num_platforms);
  platforms = (cl_platform_id *)
  malloc(sizeof(cl_platform_id)*num_platforms);
  clStatus = clGetPlatformIDs(num_platforms, platforms, NULL);
  
  //Get the devices list and choose the device you want to run on
  cl_device_id *device_list = NULL;
  cl_uint num_devices;
  clStatus = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 0,
    NULL, &num_devices);
  device_list = (cl_device_id *)
  malloc(sizeof(cl_device_id)*num_devices);
  clStatus = clGetDeviceIDs(platforms[0],
  CL_DEVICE_TYPE_CPU, num_devices, device_list, NULL);

  // Create one OpenCL context for each device in the platform
  cl_context context;
  context = clCreateContext(NULL, num_devices, device_list,
    NULL, NULL, &clStatus);

  // Create a command queue
  cl_command_queue queue = clCreateCommandQueue(
    context, device_list[0], 0, &clStatus);

  // Create the compute program from the source buffer
  cl_program program = clCreateProgramWithSource(context, 1,
                                      (const char **) & kernelSource, NULL, &err);
  // Build the program executable
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  // Create the compute kernel in the program we wish to run
  cl_kernel kernel = clCreateKernel(program, "vecAdd", &err);

  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                             bytes, h_a, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                              bytes, h_b, 0, NULL, NULL);

  // Set the arguments to our compute kernel
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

  // Execute the kernel over the entire range of the data set
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                               0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                      bytes, h_c, 0, NULL, NULL );

  //Sum up vector c and print result divided by n, this should equal 1 within error
  double sum = 0;
  for(i=0; i<n; i++)
    sum += h_c[i];
  printf("final result: %f\n", sum/n);

  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  //release host memory
  free(h_a);
  free(h_b);
  free(h_c);
  //*/
  return 0;

}

int main() {
  OpenCL_test_00();
}
