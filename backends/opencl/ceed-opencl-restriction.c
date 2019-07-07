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
  dbg("[CeedElemRestrictApply][OpenCL]");
  const cl_mem d_u;
  cl_mem d_v;
  ierr = CeedVectorGetArrayRead(u, CEED_MEM_DEVICE, (const CeedScalar**)&d_u); CeedChk(ierr);
  ierr = CeedVectorGetArray(v, CEED_MEM_DEVICE, (CeedScalar **)&d_v); CeedChk(ierr);

  cl_kernel kernel;
  if (tmode == CEED_NOTRANSPOSE) {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel=impl->noTrNoTr;
      printf("dbg: kernel=%p\n",impl->noTrNoTr);
    } else {
      kernel=impl->noTrTr;
    }
  } else {
    if (lmode == CEED_NOTRANSPOSE) {
      kernel=impl->trNoTr;
    } else {
      kernel=impl->trTr;
    }
  }

  CeedInt nelem;
  CeedElemRestrictionGetNumElements(r, &nelem);
  int nparam=3;
  size_t size2=sizeof(cl_mem);
  void *args[] = {&nparam,&size2,(void*)&d_u,&size2,(void *)&d_v,&size2,(void*)&impl->d_ind};
  dbg("[CeedElemRestrictApply][OpenCL] run_kernel");
  ierr = run_kernel(ceed,kernel,&impl->kernel_work,args); CeedChk(ierr);
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  dbg("[CeedElemRestrictApply][OpenCL] run_kernel");

  ierr = CeedVectorRestoreArrayRead(u, (const CeedScalar **)&d_u); CeedChk(ierr);
  ierr = CeedVectorRestoreArray(v, (CeedScalar **)&d_v); CeedChk(ierr);
  dbg("[CeedElemRestrictApply][OpenCL]");
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
  dbg("[CeedElemRestrictCreate][OpenCL]");
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

  CeedInt ncomp, ndof;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = compile(ceed, impl, "CeedRestrict", 4,
                 "nelem", nelem,
                 "elemsize", elemsize,
                 "ncomp", ncomp,
                 "ndof", ndof); CeedChk(ierr);
  // Architecture specific
  int nn=nelem*elemsize*ncomp;
  size_t local_size=nn<32?nn:32;
  size_t global_size=((nn+local_size-1)/local_size)*local_size;

  impl->kernel_work.work_dim=1;
  impl->kernel_work.local_work_size=local_size;
  impl->kernel_work.global_work_size=global_size;

  ierr = CeedElemRestrictionSetData(r, (void *)&impl); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_OpenCL); CeedChk(ierr);
  dbg("[CeedElemRestrictCreate][OpenCL]");
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
