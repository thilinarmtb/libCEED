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

#include <math.h>
// *****************************************************************************
// * functions for the 'no-operator' case
// *****************************************************************************
int CeedQFunctionAllocNoOpIn_OpenCL(CeedQFunction, CeedInt, CeedInt*, CeedInt*);
int CeedQFunctionAllocNoOpOut_OpenCL(CeedQFunction, CeedInt, CeedInt*, CeedInt*) ;
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
  const Ceed ceed = qf->ceed;
  CeedQFunction_OpenCL *data=qf->data;
  const Ceed_OpenCL *ceed_data=qf->ceed->data;
  const bool ocl = ceed_data->ocl;
  assert(ceed_data);

  dbg("[CeedQFunction][BuildKernel] nc=%d",data->nc);
  dbg("[CeedQFunction][BuildKernel] dim=%d",data->dim);
  dbg("[CeedQFunction][BuildKernel] nelem=%d",data->nelem);
  dbg("[CeedQFunction][BuildKernel] elemsize=%d",data->elemsize);

  char compileOptions[BUFSIZ];
  char tmp[BUFSIZ];

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
  data->program = clCreateProgramWithSource(ceed_data->context, 1, OpenCLKernels, NULL, &err);
  clBuildProgram(data->program, 1, &ceed_data->device_id, compileOptions, NULL, NULL);
  data->kQFunctionApply = clCreateKernel(data->program, "QFunctionApply", &err);

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
                                   const CeedScalar *const *in,
                                   CeedScalar *const *out) {
  const Ceed ceed = qf->ceed;
  dbg("[CeedQFunction][Apply]");
  CeedQFunction_OpenCL *data = qf->data;
  const bool from_operator_apply = data->op;
  Ceed_OpenCL *ceed_data = qf->ceed->data;

  const CeedInt bytes = sizeof(CeedScalar);
  const CeedInt ready =  data->ready;
  const CeedInt cbytes = qf->ctxsize;
  assert((Q%qf->vlength)==0); // Q must be a multiple of vlength
  const CeedInt nelem = 1; // !?
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
  //if (cbytes>0) occaCopyPtrToMem(d_ctx,qf->ctx,cbytes,0,NO_PROPS);
  if (cbytes>0) clEnqueueWriteBuffer(ceed_data->queue, d_ctx, CL_TRUE, 0,
		  cbytes, qf->ctx, 0, NULL, NULL);

  // ***************************************************************************
  dbg("[CeedQFunction][Apply] occaKernelRun");

  cl_int err;
  size_t globalSize, localSize;
  // Number of work items in each local work group
  localSize = 64;
  // Number of total work items - localSize must be devisor
  globalSize = ceil(Q/(float)localSize)*localSize;

  err  = clSetKernelArg(data->kQFunctionApply, 0, sizeof(cl_mem), &d_ctx);
  err |= clSetKernelArg(data->kQFunctionApply, 1, sizeof(CeedInt), &Q);
  err |= clSetKernelArg(data->kQFunctionApply, 2, sizeof(cl_mem), &d_idx);
  err |= clSetKernelArg(data->kQFunctionApply, 3, sizeof(cl_mem), &d_odx);
  err |= clSetKernelArg(data->kQFunctionApply, 4, sizeof(cl_mem), &d_indata);
  err |= clSetKernelArg(data->kQFunctionApply, 5, sizeof(cl_mem), &d_outdata);

  clEnqueueNDRangeKernel(ceed_data->queue, data->kQFunctionApply, 1, NULL, &globalSize,
		  &localSize, 0, NULL, NULL);

  // ***************************************************************************
//if (cbytes>0) occaCopyMemToPtr(qf->ctx,d_ctx,cbytes,0,NO_PROPS);
  if (cbytes>0) clEnqueueReadBuffer(ceed_data->queue, d_ctx, CL_TRUE, 0,
		  cbytes, qf->ctx, 0, NULL, NULL);

  // ***************************************************************************
  const int nOut = qf->numoutputfields;
  for (CeedInt i=0; i<nOut; i++) {
    const CeedEvalMode emode = qf->outputfields[i].emode;
    const char *name = qf->outputfields[i].fieldname;
    const CeedInt ncomp = qf->outputfields[i].ncomp;
    const CeedInt dim = data->dim;
    switch (emode) {
    case CEED_EVAL_NONE:
      dbg("[CeedQFunction][Apply] out \"%s\" NONE",name);
      //occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*nelem*bytes,data->oOf7[i]*bytes,
      //                 NO_PROPS);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0, Q*ncomp*nelem*bytes,
		      out[i], 0, NULL, NULL);
      break;
    case CEED_EVAL_INTERP:
      dbg("[CeedQFunction][Apply] out \"%s\" INTERP",name);
      //occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*nelem*bytes,data->oOf7[i]*bytes,
      //                 NO_PROPS);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0, Q*ncomp*nelem*bytes,
		      out[i], 0, NULL, NULL);
      break;
    case CEED_EVAL_GRAD:
      dbg("[CeedQFunction][Apply] out \"%s\" GRAD",name);
      //occaCopyMemToPtr(out[i],d_outdata,Q*ncomp*dim*nelem*bytes,data->oOf7[i]*bytes,
      //                 NO_PROPS);
      clEnqueueReadBuffer(ceed_data->queue, d_outdata, CL_TRUE, 0, Q*ncomp*dim*nelem*bytes,
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
    //occaFree(data->d_u);
    //occaFree(data->d_v);
  }
  int ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedQFunctionCreate_OpenCL
// *****************************************************************************
int CeedQFunctionCreate_OpenCL(CeedQFunction qf) {
  const Ceed ceed = qf->ceed;
  CeedQFunction_OpenCL *data;
  int ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // Populate the CeedQFunction structure **************************************
  qf->Apply = CeedQFunctionApply_OpenCL;
  qf->Destroy = CeedQFunctionDestroy_OpenCL;
  qf->data = data;
  // Fill CeedQFunction_OpenCL struct ********************************************
  data->op = false;
  data->ready = false;
  data->nc = data->dim = 1;
  data->nelem = data->elemsize = 1;
  data->e = 0;
  // Locate last ':' character in qf->focca ************************************
  dbg("[CeedQFunction][Create] focca: %s",qf->focca);
  const char *last_colon = strrchr(qf->focca,':');
  const char *last_dot = strrchr(qf->focca,'.');
  if (!last_colon)
    return CeedError(qf->ceed, 1, "Can not find ':' in function name field!");
  if (!last_dot)
    return CeedError(qf->ceed, 1, "Can not find '.' in function name field!");
  // get the function name
  data->qFunctionName = last_colon+1;
  dbg("[CeedQFunction][Create] qFunctionName: %s",data->qFunctionName);
  // extract file base name
  const char *last_slash_pos = strrchr(qf->focca,'/');
  // if no slash has been found, revert to focca field
  const char *last_slash = last_slash_pos?last_slash_pos+1:qf->focca;
  dbg("[CeedQFunction][Create] last_slash: %s",last_slash);
  // extract c_src_file & okl_base_name
  //char *c_src_file, *okl_base_name;
  //ierr = CeedCalloc(OCCA_PATH_MAX,&okl_base_name); CeedChk(ierr);
  //ierr = CeedCalloc(OCCA_PATH_MAX,&c_src_file); CeedChk(ierr);
  //memcpy(okl_base_name,last_slash,last_dot-last_slash);
  //memcpy(c_src_file,qf->focca,last_colon-qf->focca);
  //dbg("[CeedQFunction][Create] c_src_file: %s",c_src_file);
  //dbg("[CeedQFunction][Create] okl_base_name: %s",okl_base_name);
  // Now fetch OKL filename ****************************************************
  //ierr = CeedOklPath_OpenCL(ceed,c_src_file, okl_base_name, &data->oklPath);
  //CeedChk(ierr);
  // free **********************************************************************
  //ierr = CeedFree(&okl_base_name); CeedChk(ierr);
  //ierr = CeedFree(&c_src_file); CeedChk(ierr);
  return 0;
}
