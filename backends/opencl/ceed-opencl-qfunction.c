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
#define CEED_DEBUG_COLOR 177
#include "ceed-opencl.h"
#include "ceed-backend.h"
#include <math.h>
// *****************************************************************************
// * functions for the 'no-operator' case
// *****************************************************************************
int CeedQFunctionAllocNoOpIn_OpenCL(CeedQFunction, CeedInt, CeedInt*, CeedInt*);
int CeedQFunctionAllocNoOpOut_OpenCL(CeedQFunction, CeedInt, CeedInt*,
                                     CeedInt*);
int CeedQFunctionFillNoOp_OpenCL(CeedQFunction, CeedInt, cl_mem,
                                 CeedInt*, CeedInt*, const CeedScalar*const*);

// *****************************************************************************
// * functions for the 'operator' case
// *****************************************************************************
int CeedQFunctionAllocOpIn_OpenCL(CeedQFunction, CeedInt, CeedInt*, CeedInt*);
int CeedQFunctionAllocOpOut_OpenCL(CeedQFunction, CeedInt, CeedInt*, CeedInt*) ;
int CeedQFunctionFillOp_OpenCL(CeedQFunction, CeedInt, cl_mem,
                               CeedInt*, CeedInt*, const CeedScalar*const*);

// *****************************************************************************
// * buildKernel
// *****************************************************************************
static int CeedQFunctionBuildKernel(CeedQFunction qf, const CeedInt Q) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_OpenCL *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  const bool ocl = ceed_data->ocl;
  assert(ceed_data);

  dbg("[CeedQFunction][BuildKernel] nc=%d",data->nc);
  dbg("[CeedQFunction][BuildKernel] dim=%d",data->dim);
  dbg("[CeedQFunction][BuildKernel] nelem=%d",data->nelem);
  dbg("[CeedQFunction][BuildKernel] elemsize=%d",data->elemsize);

  char compileOptions[BUFSIZ], tmp[BUFSIZ];

  sprintf(tmp,"-DNC=%d", data->nc);
  strcpy(compileOptions, tmp);
  sprintf(tmp,",-DDIM=%d", data->dim);
  strcat(compileOptions, tmp);
  sprintf(tmp,",-Depsilon=%lf", data->epsilon);
  strcat(compileOptions, tmp);

  // OpenCL check for this requirement
  const CeedInt q_tile_size = (Q>OPENCL_TILE_SIZE)?OPENCL_TILE_SIZE:Q;
  // OCCA+MacOS implementation need that for now
  const CeedInt tile_size = ocl?1:q_tile_size;
  sprintf(tmp, ",-DTILE_SIZE=%d", tile_size);
  strcat(compileOptions, tmp);

  dbg("[CeedQFunction][BuildKernel] compileOptions=%s", compileOptions);
  dbg("[CeedQFunction][BuildKernel] occaDeviceBuildKernel");
  dbg("[CeedQFunction][BuildKernel] name=%s",data->qFunctionName);

  cl_int err;
  data->program = clCreateProgramWithSource(ceed_data->context, 1,
                  (const char **) &OpenCLKernels, NULL, &err);
  switch(err) {
  case CL_INVALID_CONTEXT:
    return CeedError(ceed, 1, "OpenCL backend: Invalid context.");
    break;
  case CL_INVALID_VALUE:
    return CeedError(ceed, 1, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    return CeedError(ceed, 1, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  err = clBuildProgram(data->program, 1, &ceed_data->device_id, NULL, NULL,
                       NULL);
  // Determine the size of the log
  size_t log_size;
  char *log;
  switch(err) {
  case CL_INVALID_PROGRAM:
    return CeedError(ceed, 1, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_VALUE:
    return CeedError(ceed, 1, "OpenCL backend: Invalid value.");
    break;
  case CL_INVALID_DEVICE:
    return CeedError(ceed, 1, "OpenCL backend: Invalid device.");
    break;
  case CL_INVALID_BINARY:
    return CeedError(ceed, 1, "OpenCL backend: Invalid binary.");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    return CeedError(ceed, 1, "OpenCL backend: Invalid build options.");
    break;
  case CL_INVALID_OPERATION:
    return CeedError(ceed, 1, "OpenCL backend: Invalid operation.");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    return CeedError(ceed, 1, "OpenCL backend: Compiler not available.");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    clGetProgramBuildInfo(data->program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          0, NULL,
                          &log_size);
    // Allocate memory for the log
    log = (char *) malloc(log_size);
    // Get the log
    clGetProgramBuildInfo(data->program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          log_size,
                          log, NULL);
    // Print the log
    printf("%s\n", log);
    return CeedError(ceed, 1, "OpenCL backend: Build program failure.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    return CeedError(ceed, 1, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  data->kQFunctionApply = clCreateKernel(data->program, data->qFunctionName,
                                         &err);
  switch(err) {
  case CL_INVALID_PROGRAM:
    return CeedError(ceed, 1, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    return CeedError(ceed, 1, "OpenCL backend: Invalid program executable.");
    break;
  case CL_INVALID_KERNEL_NAME:
    return CeedError(ceed, 1, "OpenCL backend: Invalid kernel name.");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    return CeedError(ceed, 1, "OpenCL backend: Invalid kernel definition.");
    break;
  case CL_INVALID_VALUE:
    return CeedError(ceed, 1, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    return CeedError(ceed, 1, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  return 0;
}

// *****************************************************************************
// * Q-functions: Apply, Destroy & Create
// *****************************************************************************
// * CEED_EVAL_NONE, no action
// * CEED_EVAL_INTERP: Q*ncomp*nelem
// * CEED_EVAL_GRAD: Q*ncomp*dim*nelem
// * CEED_EVAL_WEIGHT: Q
// *****************************************************************************
static int CeedQFunctionApply_OpenCL(CeedQFunction qf, CeedInt Q,
                                     CeedVector *In, CeedVector *Out) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  dbg("[CeedQFunction][Apply]");
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data);
  CeedQFunction_OpenCL *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  const bool from_operator_apply = data->op;
  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt ready =  data->ready;
  size_t cbytes;
  CeedInt vlength;
  ierr = CeedQFunctionGetContextSize(qf, &cbytes); CeedChk(ierr);
  ierr = CeedQFunctionGetVectorLength(qf, &vlength); CeedChk(ierr);
  assert((Q%vlength)==0); // Q must be a multiple of vlength
  const CeedInt nelem = 1; // !?
  CeedInt nIn, nOut;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, &nOut); CeedChk(ierr);
  const CeedScalar *in[16];
  CeedScalar *out[16];
  for (int i = 0; i < nIn; i++) {
    ierr = CeedVectorGetArrayRead(In[i], CEED_MEM_HOST, &in[i]); CeedChk(ierr);
  }
  for (int i = 0; i < nOut; i++) {
    ierr = CeedVectorGetArray(Out[i], CEED_MEM_HOST, &out[i]); CeedChk(ierr);
  }
  // ***************************************************************************
  if (!ready) { // If the kernel has not been built, do it now
    data->ready=true;
    CeedQFunctionBuildKernel(qf,Q);
    if (!from_operator_apply) { // like coming directly from t20-qfunction
      dbg("[CeedQFunction][Apply] NO operator_setup");
      CeedQFunctionAllocNoOpIn_OpenCL(qf,Q,&data->idx,data->iOf7);
      CeedQFunctionAllocNoOpOut_OpenCL(qf,Q,&data->odx,data->oOf7);
    } else { // coming from operator_apply
      CeedQFunctionAllocOpIn_OpenCL(qf,Q,&data->idx,data->iOf7);
      CeedQFunctionAllocOpOut_OpenCL(qf,Q,&data->odx,data->oOf7);
    }
  }
  const cl_mem d_indata = data->o_indata;
  const cl_mem d_outdata = data->o_outdata;
  const cl_mem d_ctx = data->d_ctx;
  const cl_mem d_idx = data->d_idx;
  const cl_mem d_odx = data->d_odx;
  // ***************************************************************************
  if (!from_operator_apply) {
    CeedQFunctionFillNoOp_OpenCL(qf,Q,d_indata,data->iOf7,data->oOf7,in);
  } else {
    dbg("[CeedQFunction][Apply] Operator setup, filling");
    CeedQFunctionFillOp_OpenCL(qf,Q,d_indata,data->iOf7,data->oOf7,in);
  }

  // ***************************************************************************
  void *ctx;
  if (cbytes>0) {
    ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
    clEnqueueWriteBuffer(ceed_data->queue, d_ctx, CL_TRUE, 0,
                         cbytes, qf->ctx, 0, NULL, NULL);
  }

  // ***************************************************************************
  dbg("[CeedQFunction][Apply] OpenCLKernelRun");

  cl_int err;
  size_t globalSize, localSize;
  // Number of work items in each local work group
  localSize = 1;
  // Number of total work items - localSize must be devisor
  globalSize = ceil(Q/(float)localSize)*localSize;

  err  = clSetKernelArg(data->kQFunctionApply, 0, sizeof(cl_mem), &d_ctx);
  err |= clSetKernelArg(data->kQFunctionApply, 1, sizeof(CeedInt), &Q);
  err |= clSetKernelArg(data->kQFunctionApply, 2, sizeof(cl_mem), &d_idx);
  err |= clSetKernelArg(data->kQFunctionApply, 3, sizeof(cl_mem), &d_odx);
  err |= clSetKernelArg(data->kQFunctionApply, 4, sizeof(cl_mem), &d_indata);
  err |= clSetKernelArg(data->kQFunctionApply, 5, sizeof(cl_mem), &d_outdata);

  clEnqueueNDRangeKernel(ceed_data->queue, data->kQFunctionApply, 1, NULL,
                         &globalSize,
                         &localSize, 0, NULL, NULL);

  // ***************************************************************************
  if (cbytes>0) clEnqueueReadBuffer(ceed_data->queue, d_ctx, CL_TRUE, 0,
                                      cbytes, qf->ctx, 0, NULL, NULL);

  // ***************************************************************************
  CeedQFunctionField *outputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &outputfields); CeedChk(ierr);
  for (CeedInt i=0; i<nOut; i++) {
    char *name;
    ierr = CeedQFunctionFieldGetName(outputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetNumComponents(outputfields[i], &ncomp);
    CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(outputfields[i], &emode); CeedChk(ierr);
    const CeedInt dim = data->dim;
    switch (emode) {
    case CEED_EVAL_NONE:
      dbg("[CeedQFunction][Apply] out \"%s\" NONE",name);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0,
                          Q*ncomp*nelem*bytes,
                          out[i], 0, NULL, NULL);
      break;
    case CEED_EVAL_INTERP:
      dbg("[CeedQFunction][Apply] out \"%s\" INTERP",name);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0,
                          Q*ncomp*nelem*bytes,
                          out[i], 0, NULL, NULL);
      break;
    case CEED_EVAL_GRAD:
      dbg("[CeedQFunction][Apply] out \"%s\" GRAD",name);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0,
                          Q*ncomp*dim*nelem*bytes,
                          out[i], 0, NULL, NULL);
      break;
    case CEED_EVAL_WEIGHT:
      break; // no action
    case CEED_EVAL_CURL:
      break; // Not implimented
    case CEED_EVAL_DIV:
      break; // Not implimented
    }
  }
  for (int i = 0; i < nIn; i++) {
    ierr = CeedVectorRestoreArrayRead(In[i], &in[i]); CeedChk(ierr);
  }
  for (int i = 0; i < nOut; i++) {
    ierr = CeedVectorRestoreArray(Out[i], &out[i]); CeedChk(ierr);
  }
  return 0;
}

// *****************************************************************************
// * CeedQFunctionDestroy_OpenCL
// *****************************************************************************
static int CeedQFunctionDestroy_OpenCL(CeedQFunction qf) {
  const Ceed ceed = qf->ceed;
  CeedQFunction_OpenCL *data=qf->data;
  const bool operator_setup = data->op;
  dbg("[CeedQFunction][Destroy]");
  clReleaseKernel(data->kQFunctionApply);
  clReleaseProgram(data->program);
  if (data->ready) {
    if (!operator_setup) {
      clReleaseMemObject(data->d_ctx);
      clReleaseMemObject(data->o_indata);
      clReleaseMemObject(data->o_outdata);
    }
    //clReleaseMemObject(data->d_u);
    //clReleaseMemObject(data->d_v);
  }
  int ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedQFunctionCreate_OpenCL
// *****************************************************************************
int CeedQFunctionCreate_OpenCL(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_OpenCL *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // Populate the CeedQFunction structure **************************************
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_OpenCL); CeedChk(ierr);
  // Fill CeedQFunction_OpenCL struct ********************************************
  data->op = false;
  data->ready = false;
  data->nc = data->dim = 1;
  data->nelem = data->elemsize = 1;
  data->e = 0;
  ierr = CeedQFunctionSetData(qf, (void *)&data); CeedChk(ierr);
  // Locate last ':' character in qf->focca ************************************
  char *focca;
  ierr = CeedQFunctionGetFOCCA(qf, &focca); CeedChk(ierr);
  dbg("[CeedQFunction][Create] focca: %s",focca);
  const char *last_colon = strrchr(focca,':');
  if (!last_colon)
    return CeedError(ceed, 1, "Can not find ':' in focca field!");
  // get the function name
  data->qFunctionName = last_colon+1;
  dbg("[CeedQFunction][Create] qFunctionName: %s",data->qFunctionName);
  return 0;
}
