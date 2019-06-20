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
#ifndef _ceed_opencl_h
#define _ceed_opencl_h

#include <ceed-impl.h>
#include "../include/ceed.h"
#include <ceed-backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define CeedChk_OCL(ceed, x) \
do { \
  cl_int result = (x); \
  switch(result) { \
    case CL_INVALID_COMMAND_QUEUE: \
      fprintf(stderr, "Command queue is not valid.\n"); \
      break; \
    default: \
      break; \
  } \
} while (0)

#define CEED_OPENCL_TILE_SIZE 32
#define CEED_OPENCL_MAX_BUF 3000000
#define CEED_OPENCL_NUM_RESTRICTION_KERNEL 6
// *****************************************************************************
// CEED_DEBUG_COLOR default value, forward CeedDebug* declarations & dbgcl macros
// *****************************************************************************
#ifndef CEED_DEBUG_COLOR
#define CEED_DEBUG_COLOR 0
#endif
void CeedDebugImpl_OpenCL(const Ceed,const char*,...);
void CeedDebugImpl256_OpenCL(const Ceed,const unsigned char,const char*,...);
#define CeedDebug(ceed,format, ...) CeedDebugImpl_OpenCL(ceed,format, ## __VA_ARGS__)
#define CeedDebug256(ceed,color, ...) CeedDebugImpl256_OpenCL(ceed,color, ## __VA_ARGS__)
#define dbg(...) CeedDebug256(ceed,(unsigned char)CEED_DEBUG_COLOR, ## __VA_ARGS__)

// *****************************************************************************
// Data structs for CEED OpenCL backend
// *****************************************************************************
typedef struct {
  cl_uint work_dim;
  size_t *global_work_size;
  size_t *local_work_size;
} CeedWork_OpenCL;

typedef enum {HOST_SYNC, DEVICE_SYNC, BOTH_SYNC, NONE_SYNC} SyncState;

typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  cl_mem d_array;
  cl_mem d_array_allocated;
  SyncState memState;
} CeedVector_OpenCL;

typedef struct {
  cl_kernel noTrNoTr;
  CeedWork_OpenCL *noTrNoTr_work;
  cl_kernel noTrTr;
  CeedWork_OpenCL *noTrTr_work;
  cl_kernel trNoTr;
  CeedWork_OpenCL *trNoTr_work;
  cl_kernel trTr;
  CeedWork_OpenCL *trTr_work;
  cl_kernel trNoTrIdentity;
  cl_kernel trTrIdentity;
  CeedInt *h_ind;
  CeedInt *h_ind_allocated;
  cl_mem d_ind;
  cl_mem d_ind_allocated;
} CeedElemRestriction_OpenCL;

// We use a struct to avoid having to memCpy the array of pointers
// __global__ copies by value the struct.
typedef struct {
  const CeedScalar *inputs[16];
  CeedScalar *outputs[16];
} Fields_OpenCL;

typedef struct {
  char *qFunctionName;
  cl_kernel qFunction;
  CeedWork_OpenCL *qFunction_work;
  Fields_OpenCL fields;
  cl_mem d_c;
} CeedQFunction_OpenCL;

typedef struct {
  cl_kernel interp;
  CeedWork_OpenCL *interp_work;
  cl_kernel grad;
  CeedWork_OpenCL *grad_work;
  cl_kernel weight;
  CeedWork_OpenCL *weight_work;
  cl_mem d_interp1d;
  cl_mem d_grad1d;
  cl_mem d_qweight1d;
  CeedScalar *c_B;
  CeedScalar *c_G;
} CeedBasis_OpenCL;

typedef struct {
  CeedVector
  *evecs;   /// E-vectors needed to apply operator (input followed by outputs)
  CeedScalar **edata;
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_OpenCL;

typedef struct {
  bool debug;
  char *arch;
  char *openclBackendDir;
  cl_platform_id cpPlatform;     // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  bool gpu;
  bool cpu;
} Ceed_OpenCL;

// *****************************************************************************
CEED_INTERN int CeedBasisCreateTensorH1_OpenCL(CeedInt dim,
    CeedInt P1d, CeedInt Q1d,
    const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisCreateH1_OpenCL(CeedElemTopology topo,
    CeedInt dim, CeedInt ndof, CeedInt nqpts,
    const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

//// *****************************************************************************
//CEED_INTERN int CeedElemRestrictionCreate_OpenCL(const CeedMemType mtype,
//    const CeedCopyMode cmode, const CeedInt *indices,
//    const CeedElemRestriction res);
//
//// *****************************************************************************
//CEED_INTERN int CeedElemRestrictionCreateBlocked_OpenCL(const CeedMemType mtype,
//    const CeedCopyMode cmode, const CeedInt *indices,
//    const CeedElemRestriction res);

int compile(Ceed ceed, void *data,
    const char *type,
    int nparams, ...
);

int run_kernel(Ceed ceed,
    cl_kernel kernel,
    CeedWork_OpenCL *work,
    void **args
);

#endif
