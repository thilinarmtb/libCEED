// Copyright (c) 2017-2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory. LLNL-CODE-734707.
// All Rights reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.
#define CEED_DEBUG_COLOR 10
#include "ceed-opencl.h"
#include "ceed-backend.h"
#include "unistd.h"
// *****************************************************************************
// * Callback function for OpenCL
// *****************************************************************************
void pfn_notify(const char *errinfo, const void *private_info, size_t cb,
                void *user_data) {
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, errinfo);
  fflush(stderr);
}
//*****************************************************************************
// * CeedError_OpenCL
// *****************************************************************************
static int CeedError_OpenCL(Ceed ceed,
                            const char *file, int line,
                            const char *func, int code,
                            const char *format, va_list args) {
  fprintf(stderr, "CEED-OpenCL error @ %s:%d %s\n", file, line, func);
  vfprintf(stderr, format, args);
  fprintf(stderr,"\n");
  fflush(stderr);
  abort();
  return code;
}

// *****************************************************************************
// * CeedDestroy_OpenCL
// *****************************************************************************
static int CeedDestroy_OpenCL(Ceed ceed) {
  int ierr;
  Ceed_OpenCL *data=ceed->data;
  dbg("[CeedDestroy]");

  clReleaseContext(data->context);
  clReleaseCommandQueue(data->queue);
  if(data->arch)
    free(data->arch);
  if(data->openclBackendDir)
    free(data->openclBackendDir);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedDebugImpl256
// *****************************************************************************
void CeedDebugImpl256_OpenCL(const Ceed ceed,
                             const unsigned char color,
                             const char *format,...) {
  const Ceed_OpenCL *data;
  CeedGetData(ceed, (void *)&data);
  if (!data->debug) return;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout,"\033[38;5;%dm",color);
  vfprintf(stdout,format,args);
  fprintf(stdout,"\033[m");
  fprintf(stdout,"\n");
  fflush(stdout);
  va_end(args);
}

// *****************************************************************************
// * CeedDebugImpl
// *****************************************************************************
void CeedDebugImpl_OpenCL(const Ceed ceed,
                          const char *format,...) {
  const Ceed_OpenCL *data;
  CeedGetData(ceed, (void *)&data);
  if (!data->debug) return;
  va_list args;
  va_start(args, format);
  CeedDebugImpl256_OpenCL(ceed,0,format,args);
  va_end(args);
}


// *****************************************************************************
// * INIT
// *****************************************************************************
static int CeedInit_OpenCL(const char *resource, Ceed ceed) {
  int ierr;
  Ceed_OpenCL *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ceed->data = data;

  int nrc = strlen("/cpu/opencl"); // number of characters in resource
  const bool cpu = data->cpu = !strncmp(resource, "/cpu/opencl", nrc);
  nrc = strlen("/gpu/opencl"); // number of characters in resource
  const bool gpu = data->gpu = !strncmp(resource, "/gpu/opencl", nrc);

  char *opencl = "opencl";
  char *lastSlash = strrchr(resource,'/');
  if (!strncmp(lastSlash + 1,"opencl", strlen(opencl))) {
    data->arch = NULL;
  } else {
    int archLen = resource + strlen(resource) - lastSlash;
    data->arch = calloc(sizeof(char), archLen);
    strncpy(data->arch, lastSlash+1, archLen);
  }

  // Warning: "backend cannot use resource" is used to grep in test/tap.sh
  if (!cpu && !gpu)
    return CeedError(ceed, 1, "OpenCL backend cannot use resource: %s", resource);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Error",
                                CeedError_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "Destroy",
                                CeedDestroy_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "VecCreate",
                                CeedVectorCreate_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateTensorH1",
                                CeedBasisCreateTensorH1_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "BasisCreateH1",
                                CeedBasisCreateH1_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "ElemRestrictionCreate",
                                CeedElemRestrictionCreate_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed,
                                "ElemRestrictionCreateBlocked",
                                CeedElemRestrictionCreateBlocked_OpenCL);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_OpenCL); CeedChk(ierr);

  // push env variables CEED_DEBUG or DBG to our data
  data->debug=!!getenv("CEED_DEBUG") || !!getenv("DBG");

  // Now that we can dbg, output resource and deviceID
  dbg("[CeedInit][OpenCL] resource: %s", resource);
  dbg("[CeedInit][OpenCL] data->arch = %s", data->arch);

  // Get the libceed directory
  char *subDir = "/backends/opencl/";
  const char *LIBCEED_DIR=getenv("LIBCEED_DIR");
  if (!LIBCEED_DIR) return CeedError(ceed, 1,
                                       "Cannot find LIBCEED_DIR env variable.");

  int totalSize = strlen(LIBCEED_DIR) + strlen(subDir);
  char *openclBackendDir = calloc(sizeof(char), totalSize + 1);

  strncpy(openclBackendDir, LIBCEED_DIR, strlen(LIBCEED_DIR));
  strncpy(openclBackendDir + strlen(LIBCEED_DIR), subDir, strlen(subDir));
  data->openclBackendDir = openclBackendDir;
  dbg("[CeedInit] %s", openclBackendDir);

  cl_int err;

  // Set up the Platform
  cl_platform_id* platforms = NULL;
  cl_uint num_platforms;
  err = clGetPlatformIDs(1000, NULL, &num_platforms);
  platforms = (cl_platform_id *) malloc(sizeof(cl_platform_id)*num_platforms);
  err = clGetPlatformIDs(num_platforms, platforms, NULL);

  //Get the devices list and choose the device you want to run on
  cl_device_id *device_list = NULL;
  cl_uint num_devices;
  int platformID = 1;
  if(cpu) {
    err = clGetDeviceIDs(platforms[platformID], CL_DEVICE_TYPE_CPU, 0, NULL,
                         &num_devices);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
    dbg("[CeedInit][OpenCL] CPU is selected.");
    err = clGetDeviceIDs(platforms[platformID], CL_DEVICE_TYPE_CPU, num_devices,
                         device_list, NULL);
  } else if(gpu) {
    err = clGetDeviceIDs(platforms[platformID], CL_DEVICE_TYPE_GPU, 0, NULL,
                         &num_devices);
    device_list = (cl_device_id *) malloc(sizeof(cl_device_id)*num_devices);
    dbg("[CeedInit][OpenCL] GPU is selected.");
    err = clGetDeviceIDs(platforms[platformID], CL_DEVICE_TYPE_GPU, num_devices,
                         device_list, NULL);
  }

  switch (err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_PLATFORM:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Invalid Platform");
    break;
  case CL_INVALID_DEVICE_TYPE:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Invalid Device Type");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't initialize the CPUs.: Invalid Value");
    break;
  case CL_DEVICE_NOT_FOUND:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Device not found");
    break;
  default:
    CeedError(ceed, 1, "OpenCL backend can't initialize the CPUs.: Unknown");
    break;
  }

  data->device_id = device_list[0];
  data->cpPlatform = platforms[platformID];

  cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, platforms[platformID], 0 };
  data->context = clCreateContext(properties, 1, device_list, NULL, NULL, &err);
  //data->context = clCreateContext(NULL, 1, device_list, NULL, NULL, &err);
  switch (err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_PLATFORM:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Invalid Platform");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't initialize the CPUs.: Invalid Value");
    break;
  case CL_INVALID_DEVICE:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Invalid Device");
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Device not available");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Out of host memory");
    break;
  default:
    CeedError(ceed, 1, "OpenCL backend can't initialize the CPUs.: Unknown");
    break;
  }

  cl_uint value;
  err = clGetContextInfo(data->context, CL_CONTEXT_NUM_DEVICES, sizeof(cl_uint),
                         &value, NULL);
  switch (err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_CONTEXT:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Invalid Context");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't initialize the CPUs.: Invalid Value");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Out of host memory");
    break;
  case CL_OUT_OF_RESOURCES:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the CPUs.: Out of resources");
    break;
  }
  //printf("Num devices: %u\n", value);

  data->queue = clCreateCommandQueueWithProperties(data->context, device_list[0],
                0, &err);

  return 0;
}

// *****************************************************************************
// * String concat
// *****************************************************************************
void concat(char **result, const char *s1, const char *s2) {
  *result = (char *) calloc(sizeof(char), strlen(s1) + strlen(s2) + 1);
  strcpy(*result, s1);
  strcpy(*result + strlen(s1), s2);
  //printf("result = %s\n", *result);
}

void readPythonDict(char *kernelName, CeedWork_OpenCL *work, char **kernel) {
  char buf[BUFSIZ];
  char dictName[BUFSIZ];
  int kernel_length;

  sprintf(dictName, "%s.dict", kernelName);
  FILE *fp = fopen(dictName, "r");

  if (fp != NULL) {
    do {
      fgets(buf, BUFSIZ, fp);
    } while(!strstr(buf, "\[work_dim\]"));
    fgets(buf, BUFSIZ, fp);
    sscanf(buf,"%u",&work->work_dim);
    //printf("work_dim: %u\n",work->work_dim);

    work->global_work_size = (size_t *) calloc(sizeof(size_t), work->work_dim);
    work->local_work_size = (size_t *) calloc(sizeof(size_t), work->work_dim);

    fseek(fp, 0, SEEK_SET);
    do {
      fgets(buf, BUFSIZ, fp);
    } while(!strstr(buf, "\[global_work_size\]"));

    fgets(buf, BUFSIZ, fp);
    for(int i = 0; i < work->work_dim; i++) {
      sscanf(buf,"%zd\n", work->global_work_size + i);
      //printf("global_work_dim[%d]: %zd\n",i,work->global_work_size[i]);
    }

    fseek(fp, 0, SEEK_SET);
    do {
      fgets(buf, BUFSIZ, fp);
    } while(!strstr(buf, "\[local_work_size\]"));

    fgets(buf, BUFSIZ, fp);
    for(int i = 0; i < work->work_dim; i++) {
      sscanf(buf,"%zd\n", work->local_work_size + i);
      //printf("local_work_dim[%d]: %zd\n",i,work->local_work_size[i]);
    }

    fseek(fp, 0, SEEK_SET);
    do {
      fgets(buf, BUFSIZ, fp);
    } while(!strstr(buf, "\[kernel_length\]"));
    fgets(buf, BUFSIZ, fp);
    sscanf(buf,"%d",&kernel_length);
    //printf("kernel_length=%d\n", kernel_length);

    *kernel = calloc(sizeof(char), kernel_length + 1);

    fseek(fp, 0, SEEK_SET);
    do {
      fgets(buf, BUFSIZ, fp);
    } while(!strstr(buf, "\[kernel\]"));
    fread(*kernel, sizeof(char), kernel_length, fp);
    //printf("kernel=%s\n", *kernel);
  } else {
    fprintf(stderr, "Can't open the dict file\n");
  }

  fclose(fp);
}

// *****************************************************************************
// * Build from Python
// *****************************************************************************
cl_kernel createKernelFromPython(char *kernelName, char *pythonFile, char *arch,
                                 char *constantDict, Ceed_OpenCL *ceed_data,
                                 CeedWork_OpenCL **data) {
  char pythonCmd[2*BUFSIZ];
  sprintf(pythonCmd, "python %s %s %s '%s' > %s.dict", pythonFile, kernelName, arch,
          constantDict, kernelName);
  printf("[createKernelFromPython] generating %s\n", pythonCmd);
  system(pythonCmd);

  char *kernelCode;
  *data = (CeedWork_OpenCL*) calloc(sizeof(CeedWork_OpenCL), 1);
  readPythonDict(kernelName, *data, &kernelCode);

  cl_int err;
  cl_program program;
  program = clCreateProgramWithSource(ceed_data->context, 1,
                                      (const char **) &kernelCode, NULL, &err);
  switch(err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_CONTEXT:
    fprintf(stderr, "OpenCL backend: Invalid context.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  }
  free(kernelCode);

  //err = clBuildProgram(program, 1, &ceed_data->device_id, 
  //  NULL, NULL, NULL);

  err = clBuildProgram(program, 1, &ceed_data->device_id, 
    "-cl-fast-relaxed-math -cl-denorms-are-zero", NULL, NULL);

  // Determine the size of the log
  size_t log_size;
  char *log;
  switch(err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_PROGRAM:
    fprintf(stderr, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_INVALID_DEVICE:
    fprintf(stderr, "OpenCL backend: Invalid device.");
    break;
  case CL_INVALID_BINARY:
    fprintf(stderr, "OpenCL backend: Invalid binary.");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    fprintf(stderr, "OpenCL backend: Invalid build options.");
    break;
  case CL_INVALID_OPERATION:
    fprintf(stderr, "OpenCL backend: Invalid operation.");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    fprintf(stderr, "OpenCL backend: Compiler not available.");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    clGetProgramBuildInfo(program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          0, NULL,
                          &log_size);
    // Allocate memory for the log
    log = (char *) malloc(log_size);
    // Get the log
    clGetProgramBuildInfo(program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          log_size,
                          log, NULL);
    // Print the log
    //printf("%s\n", log);
    fprintf(stderr, "OpenCL backend: Build program failure.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  }

  cl_kernel kernel   = clCreateKernel(program, kernelName, &err);
  switch(err) {
  case CL_INVALID_PROGRAM:
    fprintf(stderr, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    fprintf(stderr, "OpenCL backend: Invalid program executable.");
    break;
  case CL_INVALID_KERNEL_NAME:
    fprintf(stderr, "OpenCL backend: Invalid kernel name.");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    fprintf(stderr, "OpenCL backend: Invalid kernel definition.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  return kernel;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/opencl", CeedInit_OpenCL, 20);
  CeedRegister("/gpu/opencl", CeedInit_OpenCL, 20);
}
