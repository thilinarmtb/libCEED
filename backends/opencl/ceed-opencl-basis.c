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

  // Architecture specific
  int nn=nelem*basis->P1d*basis->ncomp;
  size_t local_size=nn<32?nn:32;
  size_t global_size=((nn+local_size-1)/local_size)*local_size;

  data->interp_work.work_dim=1;
  data->interp_work.local_work_size=local_size;
  data->interp_work.global_work_size=global_size;

  nn=nelem*basis->Q1d*basis->ncomp;
  local_size=nn<32?nn:32;
  global_size=((nn+local_size-1)/local_size)*local_size;
  data->interpT_work.work_dim=1;
  data->interpT_work.local_work_size=local_size;
  data->interpT_work.global_work_size=global_size;

  nn=nelem*basis->P1d*basis->ncomp*basis->dim;
  local_size=nn<32?nn:32;
  global_size=((nn+local_size-1)/local_size)*local_size;
  data->grad_work.work_dim=1;
  data->grad_work.local_work_size=local_size;
  data->grad_work.global_work_size=global_size;

  nn=nelem*basis->Q1d*basis->ncomp*basis->dim;
  local_size=nn<32?nn:32;
  global_size=((nn+local_size-1)/local_size)*local_size;
  data->gradT_work.work_dim=1;
  data->gradT_work.local_work_size=local_size;
  data->gradT_work.global_work_size=global_size;

  nn=nelem*CeedIntPow(basis->Q1d,basis->dim);
  local_size=nn<32?nn:32;
  global_size=((nn+local_size-1)/local_size)*local_size;
  data->weight_work.work_dim=1;
  data->weight_work.local_work_size=local_size;
  data->weight_work.global_work_size=global_size;

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
  int nparam;
  size_t size1=sizeof(int);
  size_t size2=sizeof(cl_mem);
  size_t size3=sizeof(CeedInt);
  if(emode == CEED_EVAL_INTERP) {
    nparam=4;
    void *interpargs[]={&nparam,&size1,&data->d_interp1d,&size2,&d_u,&size2,
      &d_v,&size2,(void*)&nelem,&size3};
    if(transpose) {
      ierr = run_kernel(ceed,data->interp,&data->interp_work,interpargs);
    } else {
      ierr = run_kernel(ceed,data->interpT,&data->interp_work,interpargs);
    }
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_GRAD) {
    nparam=5;
    void *gradargs[]={&nparam,&size1,&data->d_interp1d,&size2,&data->d_grad1d,&size2,
      &d_u,&size2,&d_v,&size2,(void*)&nelem,&size3};
    if(transpose) {
      ierr = run_kernel(ceed,data->grad,&data->grad_work,gradargs);
    } else {
      ierr = run_kernel(ceed,data->gradT,&data->grad_work,gradargs);
    }
    CeedChk(ierr);
  } else if (emode == CEED_EVAL_WEIGHT) {
    nparam=3;
    void *weightargs[]={&nparam,&size1,(void*)&data->d_qweight1d,&size2,&d_v,&size2,
      (void*)&nelem,&size3};
    ierr = run_kernel(ceed,data->weight,&data->weight_work,weightargs);
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

  // Remove this
  data->dim = basis->dim;

  ierr = compile(basis->ceed,data,"CeedBasis",6,
                 "elemsize",CeedIntPow(basis->P1d,basis->dim),
                 "ncomp",basis->ncomp,
                 "nqpt",CeedIntPow(basis->Q1d,basis->dim),
                 "dim",basis->dim,
                 "Q1D", basis->Q1d,
                 "P1D", basis->P1d); CeedChk(ierr);

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
