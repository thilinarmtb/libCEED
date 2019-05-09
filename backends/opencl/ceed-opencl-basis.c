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
#include "../include/ceed.h"
#include "ceed-opencl.h"

int CeedBasisApply_OpenCL(CeedBasis basis, const CeedInt nelem,
                        CeedTransposeMode tmode,
                        CeedEvalMode emode, CeedVector u, CeedVector v) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  CeedBasis_OpenCL *data;
  CeedBasisGetData(basis, (void *)&data); CeedChk(ierr);

  const CeedInt transpose = tmode == CEED_TRANSPOSE;

  cl_mem d_u;
  cl_mem d_v;
  if(emode!=CEED_EVAL_WEIGHT) {
    ierr=CeedVectorGetArrayRead(u,CEED_MEM_DEVICE,(const CeedScalar**)&d_u); CeedChk(ierr);
  }
  ierr=CeedVectorGetArray(v,CEED_MEM_DEVICE,(CeedScalar**)&d_v); CeedChk(ierr);

  if (tmode == CEED_TRANSPOSE) {
    cl_double zero=0;
    assert(sizeof(CeedScalar) == sizeof(cl_double));
    ierr = clEnqueueFillBuffer(ceed_data->queue,d_v,&zero,sizeof(cl_double),0,
        v->length*sizeof(CeedScalar),0,NULL,NULL);
  }
  if (emode == CEED_EVAL_INTERP) {
    void *interpargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d, &d_u, &d_v};
    ierr = run_kernel(ceed, data->interp, data->interp_work, interpargs);
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_GRAD) {
    void *gradargs[] = {(void *) &nelem, (void *) &transpose, &data->d_interp1d, &data->d_grad1d, &d_u, &d_v};
    ierr = run_kernel(ceed, data->grad, data->grad_work, gradargs); CeedChk(ierr);
  } else if (emode == CEED_EVAL_WEIGHT) {
    void *weightargs[] = {(void *) &nelem, (void *) &data->d_qweight1d, &d_v};
    ierr = run_kernel(ceed, data->weight, data->weight_work, weightargs);
    CeedChk(ierr);
  }

  if(emode!=CEED_EVAL_WEIGHT) {
    ierr = CeedVectorRestoreArrayRead(u, (const CeedScalar **)&d_u); CeedChk(ierr);
  }
  ierr = CeedVectorRestoreArray(v, (CeedScalar**)&d_v); CeedChk(ierr);

  return 0;
}

static int CeedBasisDestroy_OpenCL(CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);

  CeedBasis_OpenCL *data;
  ierr = CeedBasisGetData(basis, (void *) &data); CeedChk(ierr);

  ierr = clReleaseMemObject(data->d_qweight1d); CeedChk_OCL(ceed,ierr);
  ierr = clReleaseMemObject(data->d_interp1d); CeedChk_OCL(ceed,ierr);
  ierr = clReleaseMemObject(data->d_grad1d); CeedChk_OCL(ceed,ierr);

  ierr = CeedFree(&data); CeedChk(ierr);

  return 0;
}

int CeedBasisCreateTensorH1_OpenCL(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  CeedBasis_OpenCL *data;
  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void*)&ceed_data);

  const CeedInt qBytes = basis->Q1d * sizeof(CeedScalar);
  data->d_qweight1d=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,qBytes,NULL,NULL);
  CeedChk_OCL(ceed,ierr);
  clEnqueueWriteBuffer(ceed_data->queue,data->d_qweight1d,CL_TRUE,0,qBytes,basis->qweight1d,0,
      NULL,NULL);

  const CeedInt iBytes = qBytes * basis->P1d;
  data->d_interp1d=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,iBytes,NULL,NULL);
  CeedChk_OCL(ceed,ierr);
  clEnqueueWriteBuffer(ceed_data->queue,data->d_interp1d,CL_TRUE,0,qBytes,basis->interp1d,0,
      NULL,NULL);

  data->d_grad1d=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,iBytes,NULL,NULL);
  CeedChk_OCL(ceed,ierr);
  clEnqueueWriteBuffer(ceed_data->queue,data->d_grad1d,CL_TRUE,0,qBytes,basis->grad1d,0,
      NULL,NULL);

  ierr = compile(basis->ceed, data, "CeedBasis", 7,
                 "BASIS_Q1D", basis->Q1d,
                 "BASIS_P1D", basis->P1d,
                 "BASIS_BUF_LEN", basis->ncomp * CeedIntPow(basis->Q1d > basis->P1d ?
                     basis->Q1d : basis->P1d, basis->dim),
                 "BASIS_DIM", basis->dim,
                 "BASIS_NCOMP", basis->ncomp,
                 "BASIS_ELEMSIZE", CeedIntPow(basis->P1d, basis->dim),
                 "BASIS_NQPT", CeedIntPow(basis->Q1d, basis->dim)
                ); CeedChk(ierr);

  ierr = CeedBasisSetData(basis, (void *)&data);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Apply",
                                CeedBasisApply_OpenCL);
  CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Basis", basis, "Destroy",
                                CeedBasisDestroy_OpenCL);
  CeedChk(ierr);
  return 0;
}

int CeedBasisCreateH1_OpenCL(CeedElemTopology topo, CeedInt dim,
                           CeedInt ndof, CeedInt nqpts,
                           const CeedScalar *interp,
                           const CeedScalar *grad,
                           const CeedScalar *qref,
                           const CeedScalar *qweight,
                           CeedBasis basis) {
  int ierr;
  Ceed ceed;
  ierr = CeedBasisGetCeed(basis, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement generic H1 basis");
}
