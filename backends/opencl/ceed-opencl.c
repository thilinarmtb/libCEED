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

// *****************************************************************************
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

  ierr = CeedFree(&data->libceed_dir);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedDebugImpl256
// *****************************************************************************
void CeedDebugImpl256_OpenCL(const Ceed ceed,
                      const unsigned char color,
                      const char *format,...) {
  const Ceed_OpenCL *data=ceed->data;
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
  const Ceed_OpenCL *data=ceed->data;
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
  const int nrc = 9; // number of characters in resource
  const bool cpu = !strncmp(resource,"/cpu/opencl",nrc);
  const bool gpu = !strncmp(resource,"/gpu/opencl",nrc);
  const int rlen = strlen(resource);
  const bool slash = (rlen>nrc)?resource[nrc]=='/'?true:false:false;
  const int deviceID = slash?(rlen>nrc+1)?atoi(&resource[nrc+1]):0:0;
  // Warning: "backend cannot use resource" is used to grep in test/tap.sh
  if (!cpu && !gpu)
    return CeedError(ceed, 1, "OCCA backend cannot use resource: %s", resource);
  ceed->Error = CeedError_OpenCL;
  ceed->Destroy = CeedDestroy_OpenCL;
  ceed->VecCreate = CeedVectorCreate_OpenCL;
//ceed->ElemRestrictionCreate = CeedElemRestrictionCreate_OpenCL;
//ceed->ElemRestrictionCreateBlocked = CeedElemRestrictionCreateBlocked_OpenCL;
//ceed->BasisCreateTensorH1 = CeedBasisCreateTensorH1_OpenCL;
//ceed->BasisCreateH1 = CeedBasisCreateH1_OpenCL;
//ceed->QFunctionCreate = CeedQFunctionCreate_OpenCL;
//ceed->OperatorCreate = CeedOperatorCreate_OpenCL;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ceed->data = data;

  // push env variables CEED_DEBUG or DBG to our data
  data->debug=!!getenv("CEED_DEBUG") || !!getenv("DBG");
  // push ocl to our data, to be able to check it later for the kernels
  data->libceed_dir = NULL;
  if (data->debug) {
//    occaPropertiesSet(occaSettings(), "device/verbose", occaBool(1));
//    occaPropertiesSet(occaSettings(), "kernel/verbose", occaBool(1));
  }
  // Now that we can dbg, output resource and deviceID
  dbg("[CeedInit] resource: %s", resource);

  cl_int err;
  err = clGetPlatformIDs(1, &data->cpPlatform, NULL);
  if(cpu) {
    err = clGetDeviceIDs(data->cpPlatform, CL_DEVICE_TYPE_CPU, 1,&data->device_id, NULL);
  } else if(gpu) {
    err = clGetDeviceIDs(data->cpPlatform, CL_DEVICE_TYPE_GPU, 1,&data->device_id, NULL);
  }

  data->context = clCreateContext(0, 1, &data->device_id, NULL, NULL, &err);
  data->queue = clCreateCommandQueueWithProperties(data->context, data->device_id, 0, &err);

//dbg("[CeedInit] deviceID: %d", data->device_id);

//char mode[CEED_MAX_RESOURCE_LEN] = {0};
//// Push deviceID for CUDA and OpenCL mode
//if (ocl || gpu) sprintf(mode,mode_format,deviceID);
//else memcpy(mode,mode_format,strlen(mode_format));
//dbg("[CeedInit] mode: %s", mode);
//// Now creating OCCA device
//data->device = occaCreateDevice(occaString(mode));
//const char *deviceMode = occaDeviceMode(data->device);
//dbg("[CeedInit] returned deviceMode: %s", deviceMode);
//// Warning: "OCCA backend failed" is used to grep in test/tap.sh
//if (cpu && strcmp(occaDeviceMode(data->device), "Serial"))
//  return CeedError(ceed,1, "OCCA backend failed to use Serial resource");
//if (omp && strcmp(occaDeviceMode(data->device), "OpenMP"))
//  return CeedError(ceed,1, "OCCA backend failed to use OpenMP resource");
//if (gpu && strcmp(occaDeviceMode(data->device), "CUDA"))
//  return CeedError(ceed,1, "OCCA backend failed to use CUDA resource");
//if (ocl && strcmp(occaDeviceMode(data->device), "OpenCL"))
//  return CeedError(ceed,1, "OCCA backend failed to use OpenCL resource");
//// populating our data struct with libceed_dir
//ierr = CeedOklDladdr_OpenCL(ceed); CeedChk(ierr);
//if (data->libceed_dir)
//  dbg("[CeedInit] libceed_dir: %s", data->libceed_dir);
//// populating our data struct with occa_cache_dir
//char occa_cache_home[OCCA_PATH_MAX];
//const char *HOME = getenv("HOME");
//if (!HOME) return CeedError(ceed, 1, "Cannot get env HOME");
//ierr = sprintf(occa_cache_home,"%s/.occa",HOME); CeedChk(!ierr);
//const char *OCCA_CACHE_DIR = getenv("OCCA_CACHE_DIR");
//const char *occa_cache_dir = OCCA_CACHE_DIR?OCCA_CACHE_DIR:occa_cache_home;
//const int occa_cache_dir_len = strlen(occa_cache_dir);
//ierr = CeedCalloc(occa_cache_dir_len+1,&data->occa_cache_dir); CeedChk(ierr);
//memcpy(data->occa_cache_dir,occa_cache_dir,occa_cache_dir_len+1);
//dbg("[CeedInit] occa_cache_dir: %s", data->occa_cache_dir);
  return 0;
}

// *****************************************************************************
// * REGISTER
// *****************************************************************************
__attribute__((constructor))
static void Register(void) {
  CeedRegister("/cpu/opencl", CeedInit_OpenCL, 20);
  CeedRegister("/gpu/opencl", CeedInit_OpenCL, 20);
}
