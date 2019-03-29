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
  dbg("[CeedQFunction][BuildKernel] name=%s",data->qFunctionName);

  // ***************************************************************************
  char *arch = ceed_data->arch;
  char constantDict[BUFSIZ];
  sprintf(constantDict, "{\"nc\": %d,"
          "\"dim\": %d,"
          "\"epsilon\": %lf}",
          data->nc, data->dim, 1.e-14);

  data->kQFunctionApply = createKernelFromPython(data->qFunctionName, arch,
                          constantDict, data->pythonFile,
                          ceed);
  // ***************************************************************************

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
    //for(int j = 0; j<27; j++) {
    //  printf("in[%d][%d] = %lf\n",i,j,in[i][j]);
    //}
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

  cl_double* pointer;
  cl_int* pointer1;
  cl_int* pointer2;
  cl_double *pointer3;
  
  // CHECK INPUT DATA
  //cl_double *pointer = (cl_double*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_indata, CL_TRUE, CL_MAP_READ, 0, sizeof(double), 0, NULL, NULL, NULL);
  //for(int i=0; i<81; i++) {
  //  printf("indata_from_device[%d]=%lf\n",i,pointer[i]);
  //}
  //cl_int *pointer1 = (cl_int*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_idx, CL_TRUE, CL_MAP_READ, 0, sizeof(int), 0, NULL, NULL, NULL);
  //for(int i=0; i<3; i++) {
  //  printf("idx_from_device[%d]=%d\n",i,pointer1[i]);
  //}
  //cl_int *pointer2 = (cl_int*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_odx, CL_TRUE, CL_MAP_READ, 0, sizeof(int), 0, NULL, NULL, NULL);
  //for(int i=0; i<3; i++) {
  //  printf("odx_from_device[%d]=%d\n",i,pointer2[i]);
  //}

  //clEnqueueUnmapMemObject(ceed_data->queue, d_indata, pointer, NULL, NULL, NULL);
  //clEnqueueUnmapMemObject(ceed_data->queue, d_idx, pointer1, NULL, NULL, NULL);
  //clEnqueueUnmapMemObject(ceed_data->queue, d_odx, pointer2, NULL, NULL, NULL);
  // END CHECK INPUT DATA

  // ***************************************************************************
  void *ctx;
  if (cbytes>0) {
    ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
    clEnqueueWriteBuffer(ceed_data->queue, d_ctx, CL_TRUE, 0,
                         cbytes, ctx, 0, NULL, NULL);
  }

  // ***************************************************************************
  dbg("[CeedQFunction][Apply] OpenCLKernelRun");

  cl_int err;
  size_t globalSize, localSize;
  // Number of work items in each local work group
  localSize = 1;
  // Number of total work items - localSize must be devisor
  globalSize = ceil(Q/(float)localSize)*localSize;

  err  = clSetKernelArg(data->kQFunctionApply, 0, sizeof(cl_mem), (void*)&d_ctx);
  err = clSetKernelArg(data->kQFunctionApply, 1, sizeof(CeedInt), (void*) &Q);
  err = clSetKernelArg(data->kQFunctionApply, 2, sizeof(cl_mem), (void*)&d_idx);
  err = clSetKernelArg(data->kQFunctionApply, 3, sizeof(cl_mem), (void*)&d_odx);
  err = clSetKernelArg(data->kQFunctionApply, 4, sizeof(cl_mem), (void*)&d_indata);
  err = clSetKernelArg(data->kQFunctionApply, 5, sizeof(cl_mem), (void*)&d_outdata);

  err = clEnqueueNDRangeKernel(ceed_data->queue, data->kQFunctionApply, 1, NULL,
                         &globalSize,
                         &localSize, 0, NULL, NULL);

  clFlush(ceed_data->queue);
  clFinish(ceed_data->queue);

   // CHECK INPUT DATA
  
  //pointer = (cl_double*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_indata, CL_TRUE, CL_MAP_READ, 0, 2*Q*sizeof(double), 0, NULL, NULL, &err);
  //printf("STATUS: %d\n", err);
  //for(int i=0; i<2*Q; i++) { 
  //  printf("indata_from_device[%d]=%lf\n",i,pointer[i]);
  //}
  //pointer1 = (cl_int*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_idx, CL_TRUE, CL_MAP_READ, 0, 3*sizeof(cl_int), 0, NULL, NULL, &err);
  //printf("STATUS: %d\n", err);
  //assert(pointer1[1] > 0);
  //for(int i=0; i<3; i++) {
  //  printf("idx_from_device[%d]=%d\n",i,pointer1[i]);
  //}
  //pointer2 = (cl_int*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_odx, CL_TRUE, CL_MAP_READ, 0, 3*sizeof(cl_int), 0, NULL, NULL, &err);
  //printf("STATUS: %d\n", err);
  //for(int i=0; i<3; i++) {
  //  printf("odx_from_device[%d]=%d\n",i,pointer2[i]);
  //}

  //pointer3 = (cl_double*)clEnqueueMapBuffer(ceed_data->queue,
  //    d_outdata, CL_TRUE, CL_MAP_READ, 0, 2*Q*sizeof(double), 0, NULL, NULL, &err);
  //printf("STATUS: %d\n", err);
  //for(int i=0; i<2*Q; i++) {
  //  printf("outdata_from_device[%d]=%g\n",i,pointer3[i]);
  //}

   //CHECK OUTPUT DATA
  //exit(0);

  //clEnqueueUnmapMemObject(ceed_data->queue, d_indata, pointer, NULL, NULL, NULL);
  //clEnqueueUnmapMemObject(ceed_data->queue, d_idx, pointer1, NULL, NULL, NULL);
  //clEnqueueUnmapMemObject(ceed_data->queue, d_odx, pointer2, NULL, NULL, NULL);
  //clEnqueueUnmapMemObject(ceed_data->queue, d_outdata, pointer3, NULL, NULL, NULL);
  // END CHECK INPUT DATA 
  
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

      //printf("err from Readbuffer = %d\n", err);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 
        data->oOf7[i]*bytes,Q*ncomp*nelem*bytes, out[i], 0, NULL, NULL);

      //for(int j= 0; j<Q*ncomp*nelem; j++) {
      //  printf("%s %lf\n",name, out[i][j]);
      //}
      break;
    case CEED_EVAL_INTERP:
      dbg("[CeedQFunction][Apply] out \"%s\" INTERP %d",name, Q);

      err = clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE,
        data->oOf7[i]*bytes, Q*ncomp*nelem*bytes, out[i], 0, NULL, NULL);
      //printf("err from Readbuffer = %d\n", err);

      //for(int j= 0; j<Q*ncomp*nelem; j++) {
      //  printf("%s %g\n",name, out[i][j]);
      //}
      break;
    case CEED_EVAL_GRAD:
      dbg("[CeedQFunction][Apply] out \"%s\" GRAD",name);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, data->oOf7[i]*bytes,
          Q*ncomp*dim*nelem*bytes, out[i], 0, NULL, NULL);
      //for(int j= 0; j<Q*ncomp*dim*nelem; j++) {
      //  printf("%s %lf\n",name, out[i][j]);
      //}
      break;
    case CEED_EVAL_WEIGHT:
      break; // no action
    case CEED_EVAL_CURL:
      break; // Not implimented
    case CEED_EVAL_DIV:
      break; // Not implimented
    }
  }


  //for (int i = 0; i < nOut; i++) {
  //  printf("pointer-out[%d]=%p\n",i,out[i]);
  //    for(int j= 0; j<27; j++) {
  //      printf("out[%d][%d] %lf\n",i,j, out[i][j]);
  //    }
  //}
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
  if (data->ready) {
    if (!operator_setup) {
      clReleaseMemObject(data->d_ctx);
      clReleaseMemObject(data->o_indata);
      clReleaseMemObject(data->o_outdata);
    }
    //clReleaseMemObject(data->d_u);
    //clReleaseMemObject(data->d_v);
  }

  if(data->qFunctionName)
    free(data->qFunctionName);
  if(data->pythonFile)
    free(data->pythonFile);

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
  // Locate last ':' and '.' character in qf->focca ************************************
  char *focca;
  ierr = CeedQFunctionGetFOCCA(qf, &focca); CeedChk(ierr);

  dbg("[CeedQFunction][Create] focca: %s",focca);
  const char *last_colon = strrchr(focca,':');
  if (!last_colon)
    return CeedError(ceed, 1, "Can not find ':' in focca field!");
  const char *last_dot = strrchr(focca,'.');
  if (!last_dot)
    return CeedError(ceed, 1, "Can not find '.' in focca field!");

  // get the function name
  int size = strlen(focca) - (last_colon - focca + 1);
  data->qFunctionName = (char *) calloc(sizeof(char), size + 1);
  strncpy(data->qFunctionName, last_colon+1, size);

  size = last_dot - focca;
  data->pythonFile = (char *) calloc(sizeof(char), size + 1 + 3);
  strncpy(data->pythonFile, focca, size + 1);
  data->pythonFile[size+1] = 'p';
  data->pythonFile[size+2] = 'y';

  dbg("[CeedQFunction][Create] qFunctionName: %s",data->qFunctionName);
  dbg("[CeedQFunction][Create] pythonFile: %s",data->pythonFile);

  return 0;
}
