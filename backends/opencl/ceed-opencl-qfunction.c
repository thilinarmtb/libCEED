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
#include <stdio.h>
#include "../include/ceed.h"
#include "ceed-opencl.h"

static int CeedQFunctionApply_OpenCL(CeedQFunction qf, CeedInt Q,
                                   CeedVector *U, CeedVector *V) {
  int ierr;
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);
  CeedQFunction_OpenCL *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_OpenCL;
  ierr = CeedGetData(ceed, (void *)&ceed_OpenCL);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  CeedChk(ierr);

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorGetArrayRead(U[i], CEED_MEM_DEVICE, &data->fields.inputs[i]);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorGetArray(V[i], CEED_MEM_DEVICE, &data->fields.outputs[i]);
    CeedChk(ierr);
  }

  // TODO find a way to avoid this systematic memCpy

  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);
  if (ctxsize > 0) {
    if(!data->d_c) {
      data->d_c = clCreateBuffer(ceed_OpenCL->context,CL_MEM_READ_WRITE,ctxsize,NULL,&ierr);
    }
    void *ctx;
    ierr = CeedQFunctionGetInnerContext(qf, &ctx); CeedChk(ierr);
    clEnqueueWriteBuffer(ceed_OpenCL->queue,data->d_c,CL_TRUE,0,ctxsize,ctx,0,NULL,NULL);
    CeedChk_OCL(ceed, ierr);
  }

  void *ctx;
  ierr = CeedQFunctionGetContext(qf, &ctx); CeedChk(ierr);
  // void *args[] = {&ctx, (void*)&Q, &data->d_u, &data->d_v};
  void *args[] = {&data->d_c, (void *) &Q, &data->fields};
  ierr = run_kernel(ceed,data->qFunction,&data->qFunction_work,args);
  CeedChk(ierr);

  for (CeedInt i = 0; i < numinputfields; i++) {
    ierr = CeedVectorRestoreArrayRead(U[i], &data->fields.inputs[i]);
    CeedChk(ierr);
  }

  for (CeedInt i = 0; i < numoutputfields; i++) {
    ierr = CeedVectorRestoreArray(V[i], &data->fields.outputs[i]);
    CeedChk(ierr);
  }

  return 0;
}

static int CeedQFunctionDestroy_OpenCL(CeedQFunction qf) {
  int ierr;
  CeedQFunction_OpenCL *data;
  ierr = CeedQFunctionGetData(qf, (void *)&data); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedQFunctionGetCeed(qf, &ceed); CeedChk(ierr);

  ierr = clReleaseMemObject(data->d_c); CeedChk_OCL(ceed, ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedQFunctionCreate_OpenCL(CeedQFunction qf) {
  int ierr;
  Ceed ceed;
  CeedQFunctionGetCeed(qf, &ceed);
  CeedQFunction_OpenCL *data;
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  ierr = CeedQFunctionSetData(qf, (void *)&data); CeedChk(ierr);
  CeedInt numinputfields, numoutputfields;
  ierr = CeedQFunctionGetNumArgs(qf, &numinputfields, &numoutputfields);
  size_t ctxsize;
  ierr = CeedQFunctionGetContextSize(qf, &ctxsize); CeedChk(ierr);

  Ceed_OpenCL *ceed_OpenCL;
  ierr = CeedGetData(ceed, (void *)&ceed_OpenCL);
  data->d_c = clCreateBuffer(ceed_OpenCL->context,CL_MEM_READ_WRITE,ctxsize,NULL,&ierr);

  const char *funname = strrchr(qf->focca, ':') + 1;
  data->qFunctionName = (char *)funname;
  const int filenamelen = funname - qf->focca;
  char filename[filenamelen];
  memcpy(filename, qf->focca, filenamelen - 1);
  filename[filenamelen - 1] = '\0';
  //TODO
  // compileQFunction: loadOpenCLFunction(qf, filename); CeedChk(ierr);

  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Apply",
                                CeedQFunctionApply_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "QFunction", qf, "Destroy",
                                CeedQFunctionDestroy_OpenCL); CeedChk(ierr);
  return 0;
}
