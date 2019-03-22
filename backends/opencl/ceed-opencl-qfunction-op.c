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
// *****************************************************************************
// * Alloc function for with operator case
// *****************************************************************************
int CeedQFunctionAllocOpIn_OpenCL(CeedQFunction qf, CeedInt Q,
                                  CeedInt *idx_p,
                                  CeedInt *iOf7) {
  int ierr;
  CeedInt idx = 0;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_OpenCL *qf_data;
  ierr = CeedQFunctionGetData(qf, (void*)&qf_data); CeedChk(ierr);
  CeedOperator op = qf_data->op;
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  CeedInt nIn;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, NULL); CeedChk(ierr);
  assert(nIn<N_MAX_IDX);
  size_t cbytes;
  ierr = CeedQFunctionGetContextSize(qf, &cbytes); CeedChk(ierr);
  const CeedInt bytes = sizeof(CeedScalar);
  // ***************************************************************************
  dbg("[CeedQFunction][AllocOpIn]");
  CeedQFunctionField *inputfields;
  ierr = CeedQFunctionGetFields(qf, &inputfields, NULL); CeedChk(ierr);
  CeedOperatorField *opfields;
  ierr = CeedOperatorGetFields(op, &opfields, NULL); CeedChk(ierr);
  for (CeedInt i=0; i<nIn; i++) {
    dbg("\t[CeedQFunction][AllocOpIn] # %d/%d",i,nIn-1);
    char *name;
    ierr = CeedQFunctionFieldGetName(inputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetNumComponents(inputfields[i], &ncomp);
    CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(inputfields[i], &emode); CeedChk(ierr);
    CeedBasis basis;
    ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
    CeedInt dim = 0;
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
    }
    switch(emode) {
    case CEED_EVAL_INTERP:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > INTERP (%d)", name,Q*ncomp);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp;
      idx+=1;
      break;
    case CEED_EVAL_GRAD:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > GRAD (%d)",name,Q*ncomp*dim);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp*dim;;
      idx+=1;
      break;
    case CEED_EVAL_NONE:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > NONE",name);
      iOf7[idx+1]=iOf7[idx]+Q*ncomp;
      idx+=1;
      break;
    case CEED_EVAL_WEIGHT:
      dbg("\t[CeedQFunction][AllocOpIn] \"%s\" > WEIGHT (%d)",name,Q);
      iOf7[idx+1]=iOf7[idx]+Q;
      idx+=1;
      break;
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  assert(idx==nIn);
  *idx_p = idx;
  const CeedInt ilen=iOf7[idx];
  dbg("[CeedQFunction][AllocOpIn] ilen=%d", ilen);
  // INPUT+IDX alloc ***********************************************************
  assert(ilen>0);
  qf_data->o_indata = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                     ilen*bytes, NULL, NULL);
  qf_data->d_idx = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                  idx*bytes, NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, qf_data->d_idx, CL_TRUE, 0,
                       idx*sizeof(int),
                       iOf7, 0, NULL, NULL);
  // CTX alloc *****************************************************************
  qf_data->d_ctx = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                  cbytes>0?cbytes:32, NULL, NULL);
  return 0;
}

// *****************************************************************************
// * Alloc function for with operator case
// *****************************************************************************
int CeedQFunctionAllocOpOut_OpenCL(CeedQFunction qf, CeedInt Q,
                                   CeedInt *odx_p,
                                   CeedInt *oOf7) {
  int ierr;
  CeedInt odx = 0;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_OpenCL *data;
  ierr = CeedQFunctionGetData(qf, (void*)&data); CeedChk(ierr);
  CeedOperator op = data->op;
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  const CeedInt bytes = sizeof(CeedScalar);
  CeedInt nOut;
  ierr = CeedQFunctionGetNumArgs(qf, NULL, &nOut); CeedChk(ierr);
  assert(nOut<N_MAX_IDX);
  dbg("\n[CeedQFunction][AllocOpOut]");
  CeedQFunctionField *outputfields;
  ierr = CeedQFunctionGetFields(qf, NULL, &outputfields); CeedChk(ierr);
  CeedOperatorField *opfields;
  ierr = CeedOperatorGetFields(op, NULL, &opfields); CeedChk(ierr);
  for (CeedInt i=0; i<nOut; i++) {
    dbg("\t[CeedQFunction][AllocOpOut] # %d/%d",i,nOut-1);
    char *name;
    ierr = CeedQFunctionFieldGetName(outputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetNumComponents(outputfields[i], &ncomp);
    CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(outputfields[i], &emode); CeedChk(ierr);
    CeedBasis basis;
    ierr = CeedOperatorFieldGetBasis(opfields[i], &basis); CeedChk(ierr);
    CeedInt dim = 0;
    if (basis != CEED_BASIS_COLLOCATED) {
      ierr = CeedBasisGetDimension(basis, &dim); CeedChk(ierr);
    }
    switch(emode) {
    case CEED_EVAL_NONE:
      dbg("[CeedQFunction][AllocOpOut] out \"%s\" NONE (%d)",name,Q*ncomp);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp;
      odx+=1;
      break;
    case CEED_EVAL_INTERP:
      dbg("\t[CeedQFunction][AllocOpOut \"%s\" > INTERP (%d)", name,Q*ncomp);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp;
      odx+=1;
      break;
    case CEED_EVAL_GRAD:
      dbg("\t[CeedQFunction][AllocOpOut] \"%s\" > GRAD (%d)",name,Q*ncomp*dim);
      oOf7[odx+1]=oOf7[odx]+Q*ncomp*dim;
      odx+=1;
      break;
    case CEED_EVAL_WEIGHT:
      break; // Should not occur
    case CEED_EVAL_DIV:
      break; // Not implimented
    case CEED_EVAL_CURL:
      break; // Not implimented
    }
  }
  for (CeedInt i=0; i<odx+1; i++) {
    dbg("\t[CeedQFunction][AllocOpOut] oOf7[%d]=%d", i,oOf7[i]);
  }
  //assert(odx==nOut);
  *odx_p = odx;
  const CeedInt olen=oOf7[odx];
  dbg("[CeedQFunction][AllocOpOut] olen=%d", olen);
  assert(olen>0);
  dbg("[CeedQFunction][AllocOpIn] Alloc OUT of length %d", olen);
  // OUTPUT alloc **********************************************************
  data->o_outdata = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                   olen*bytes, NULL, NULL);
  data->d_odx = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                               odx*sizeof(int), NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, data->d_odx, CL_TRUE, 0, odx*sizeof(int),
                       oOf7, 0, NULL, NULL);
  return 0;
}

// *****************************************************************************
// * Fill function for with operator case
// *****************************************************************************
int CeedQFunctionFillOp_OpenCL(CeedQFunction qf, CeedInt Q,
                               cl_mem d_indata,
                               CeedInt *iOf7,
                               CeedInt *oOf7,
                               const CeedScalar *const *in) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data);
  CeedInt nIn;
  ierr = CeedQFunctionGetNumArgs(qf, &nIn, NULL); CeedChk(ierr);
  const CeedInt bytes = sizeof(CeedScalar);
  dbg("\n[CeedQFunction][FillOp]");
  CeedQFunctionField *inputfields;
  ierr = CeedQFunctionGetFields(qf, &inputfields, NULL); CeedChk(ierr);
  for (CeedInt i=0; i<nIn; i++) {
    char *name;
    ierr = CeedQFunctionFieldGetName(inputfields[i], &name); CeedChk(ierr);
    CeedInt ncomp;
    ierr = CeedQFunctionFieldGetNumComponents(inputfields[i], &ncomp);
    CeedChk(ierr);
    CeedEvalMode emode;
    ierr = CeedQFunctionFieldGetEvalMode(inputfields[i], &emode); CeedChk(ierr);
    switch(emode) {
    case CEED_EVAL_NONE: {
      dbg("[CeedQFunction][FillOp] \"%s\" > NONE",name);
      const CeedInt length = Q*ncomp;
      dbg("[CeedQFunction][FillOp] NONE length=%d", length);
      dbg("[CeedQFunction][FillOp] NONE offset=%d", iOf7[i]);
      assert(length>0);
      clEnqueueWriteBuffer(ceed_data->queue, d_indata, CL_TRUE, iOf7[i]*bytes,
                           length*bytes, in[i], 0, NULL, NULL);
      break;
    }
    case CEED_EVAL_INTERP: {
      dbg("[CeedQFunction][FillOp] \"%s\" INTERP", name);
      dbg("[CeedQFunction][FillOp] INTERP iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      dbg("[CeedQFunction][FillOp] INTERP length=%d", length);
      dbg("[CeedQFunction][FillOp] INTERP offset=%d", iOf7[i]);
      assert(length>0);
      clEnqueueWriteBuffer(ceed_data->queue, d_indata, CL_TRUE, iOf7[i]*bytes,
                           length*bytes, in[i], 0, NULL, NULL);
      break;
    }
    case CEED_EVAL_GRAD: {
      dbg("[CeedQFunction][FillOp] \"%s\" GRAD", name);
      dbg("[CeedQFunction][FillOp] GRAD iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      dbg("[CeedQFunction][FillOp] GRAD length=%d", length);
      dbg("[CeedQFunction][FillOp] GRAD offset=%d", iOf7[i]);
      assert(length>0);
      clEnqueueWriteBuffer(ceed_data->queue, d_indata, CL_TRUE, iOf7[i]*bytes,
                           length*bytes, in[i], 0, NULL, NULL);
      break;
    }
    case CEED_EVAL_WEIGHT:
      dbg("[CeedQFunction][FillOp] \"%s\" WEIGHT", name);
      dbg("[CeedQFunction][FillOp] WEIGHT iOf7[%d]=%d", i,iOf7[i]);
      const CeedInt length = iOf7[i+1]-iOf7[i];
      dbg("[CeedQFunction][FillOp] WEIGHT length=%d", length);
      dbg("[CeedQFunction][FillOp] WEIGHT offset=%d", iOf7[i]);
      assert(length>0);
      clEnqueueWriteBuffer(ceed_data->queue, d_indata, CL_TRUE, iOf7[i]*bytes,
                           length*bytes, in[i], 0, NULL, NULL);
      break;  // No action
    case CEED_EVAL_DIV: break; // Not implemented
    case CEED_EVAL_CURL: break; // Not implemented
    }
  }
  return 0;
}
