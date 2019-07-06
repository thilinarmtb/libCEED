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

#include <ceed-impl.h>
#include <string.h>
#include <stdarg.h>
#include "ceed-opencl.h"
#include "unistd.h"

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
                                CeedElemRestrictionCreateBlocked_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "QFunctionCreate",
                                CeedQFunctionCreate_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Ceed", ceed, "OperatorCreate",
                                CeedOperatorCreate_OpenCL); CeedChk(ierr);

  // push env variables CEED_DEBUG or DBG to our data
  data->debug=!!getenv("CEED_DEBUG") || !!getenv("DBG");

  // Now that we can dbg, output resource and deviceID
  dbg("[CeedInit][OpenCL] resource: %s", resource);
  dbg("[CeedInit][OpenCL] data->arch = %s", data->arch);

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
  int platformID = 0;
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
              "OpenCL backend can't initialize the device: Invalid Platform");
    break;
  case CL_INVALID_DEVICE_TYPE:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the device: Invalid Device Type");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't initialize the device: Invalid Value");
    break;
  case CL_DEVICE_NOT_FOUND:
    CeedError(ceed, 1,
              "OpenCL backend can't initialize the device: Device not found");
    break;
  default:
    CeedError(ceed, 1, "OpenCL backend can't initialize the device: Unknown reason");
    break;
  }

  data->device_id = device_list[0];
  data->cpPlatform = platforms[platformID];

  cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM,(cl_context_properties)data->cpPlatform,0};
  data->context = clCreateContext(properties, 1, device_list, NULL, NULL, &err);
  switch (err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_PLATFORM:
    CeedError(ceed, 1,
              "OpenCL backend can't create context: Invalid Platform");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't create context: Invalid Value");
    break;
  case CL_INVALID_DEVICE:
    CeedError(ceed, 1,
              "OpenCL backend can't create context: Invalid Device");
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    CeedError(ceed, 1,
              "OpenCL backend can't create context: Device not available");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    CeedError(ceed, 1,
              "OpenCL backend can't create context: Out of host memory");
    break;
  default:
    CeedError(ceed, 1, "OpenCL backend can't create context: Unknown");
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
              "OpenCL backend can't get context info: Invalid Context");
    break;
  case CL_INVALID_VALUE:
    CeedError(ceed, 1, "OpenCL backend can't get context info: Invalid Value");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    CeedError(ceed, 1,
              "OpenCL backend can't get context info: Out of host memory");
    break;
  case CL_OUT_OF_RESOURCES:
    CeedError(ceed, 1,
              "OpenCL backend can't get context info: Out of resources");
    break;
  }

  data->queue = clCreateCommandQueueWithProperties(data->context, device_list[0],
                0, &err);
  ierr = CeedSetData(ceed,(void *)&data); CeedChk(ierr);
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

__attribute__((constructor))
static void Register(void) {
  init_loopy();
  CeedRegister("/cpu/opencl", CeedInit_OpenCL, 20);
  CeedRegister("/gpu/opencl", CeedInit_OpenCL, 20);
}
