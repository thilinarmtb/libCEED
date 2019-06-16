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
#include "ceed-opencl.h"

static int CeedElemRestrictionApply_OpenCL(CeedElemRestriction r,
    CeedTransposeMode tmode, CeedTransposeMode lmode,
    CeedVector u, CeedVector v, CeedRequest *request) {
  int ierr;
  CeedElemRestriction_OpenCL *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  const cl_mem d_u;
  cl_mem *d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, (const CeedScalar **)&d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, (CeedScalar **)&d_v); CeedChk(ierr);

  cl_kernel kernel;
  CeedWork_OpenCL *kernel_work;
  if (tmode == CEED_NOTRANSPOSE) {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel = impl->noTrNoTr;
      kernel_work = impl->noTrNoTr_work;
    } else {
      kernel = impl->noTrTr;
      kernel_work = impl->noTrTr_work;
    }
  } else {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel = impl->trNoTr;
      kernel_work = impl->trNoTr_work;
    } else {
      kernel = impl->trTr;
      kernel_work = impl->trTr_work;
    }
  }

  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);
  void *args[] = {&nelem,&impl->d_ind,(void *)&d_u,&d_v};
  ierr = run_kernel(ceed, kernel, kernel_work, args); CeedChk(ierr);

  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;

  ierr = CeedVectorRestoreArrayRead(u, (const CeedScalar **)&d_u); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, (CeedScalar **)&d_v); CeedChk(ierr);
  return 0;
}

static int CeedElemRestrictionDestroy_OpenCL(CeedElemRestriction r) {
  int ierr;
  CeedElemRestriction_OpenCL *impl;
  ierr = CeedElemRestrictionGetData(r, (void *)&impl); CeedChk(ierr);

  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  ierr = CeedFree(&impl->h_ind_allocated); CeedChk(ierr);
  ierr = clReleaseMemObject(impl->d_ind_allocated); CeedChk_OCL(ceed, ierr);
  ierr = CeedFree(&r->data); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreate_OpenCL(CeedMemType mtype,
                                       CeedCopyMode cmode,
                                       const CeedInt *indices,
                                       CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void *)&ceed_data); CeedChk(ierr);
  CeedElemRestriction_OpenCL *impl;
  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  CeedInt size = nelem * elemsize;

  ierr = CeedCalloc(1, &impl); CeedChk(ierr);
  impl->h_ind           = NULL;
  impl->h_ind_allocated = NULL;
  impl->d_ind           = NULL;
  impl->d_ind_allocated = NULL;

  if (mtype == CEED_MEM_HOST) {
    switch (cmode) {
    case CEED_OWN_POINTER:
      impl->h_ind_allocated = (CeedInt *)indices;
      impl->h_ind = (CeedInt *)indices;
      break;
    case CEED_USE_POINTER:
      impl->h_ind = (CeedInt *)indices;
      break;
    case CEED_COPY_VALUES:
      break;
    }
    if (indices != NULL) {
      impl->d_ind=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,
          size*sizeof(CeedInt),NULL,NULL);
      CeedChk_OCL(ceed, ierr);
      clEnqueueWriteBuffer(ceed_data->queue,impl->d_ind,CL_TRUE,0,size*sizeof(CeedInt),
          indices,0,NULL,NULL);
      impl->d_ind_allocated = impl->d_ind;//We own the device memory
    }
  } else if (mtype == CEED_MEM_DEVICE) {
    switch (cmode) {
    case CEED_COPY_VALUES:
      if (indices != NULL) {
        impl->d_ind=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,
            size*sizeof(CeedInt),NULL,NULL);
        CeedChk_OCL(ceed, ierr);
        clEnqueueWriteBuffer(ceed_data->queue,impl->d_ind,CL_TRUE,0,size*sizeof(CeedInt),
            indices,0,NULL,NULL);
        impl->d_ind_allocated = impl->d_ind;//We own the device memory
        CeedChk_OCL(ceed, ierr);
      }
      break;
    case CEED_OWN_POINTER:
      impl->d_ind = (cl_mem)indices;
      impl->d_ind_allocated = impl->d_ind;
      break;
    case CEED_USE_POINTER:
      impl->d_ind = (cl_mem)indices;
    }
  } else
    return CeedError(ceed, 1, "Only MemType = HOST or DEVICE supported");

  CeedInt ncomp, ndof, nelem;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = compile(ceed, impl, "CeedRestrict", 4,
                 "RESTRICTION_NELEM", nelem,
                 "RESTRICTION_ELEMSIZE", elemsize,
                 "RESTRICTION_NCOMP", ncomp,
                 "RESTRICTION_NDOF", ndof); CeedChk(ierr);

  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_OpenCL); CeedChk(ierr);
  return 0;
}

int CeedElemRestrictionCreateBlocked_OpenCL(const CeedMemType mtype,
                                              const CeedCopyMode cmode,
                                              const CeedInt *indices,
                                              const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
