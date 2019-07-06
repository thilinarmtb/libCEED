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
#include "ceed-opencl.h"
#include "string.h"

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  int ierr;
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
  return length * sizeof(CeedScalar);
}

static inline int CeedSyncD2H_OpenCL(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed, (void*)&ceed_data);

  cl_int err = clEnqueueWriteBuffer(ceed_data->queue,
      data->d_array, CL_TRUE, 0, bytes(vec), data->h_array, 0, NULL, NULL);
  CeedChk_OCL(ceed,err);
  return 0;
}

static inline int CeedSyncH2D_OpenCL(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed, (void*)&ceed_data);

  cl_int err = clEnqueueReadBuffer(ceed_data->queue,
      data->d_array, CL_TRUE, 0, bytes(vec), data->h_array, 0, NULL, NULL);
  CeedChk_OCL(ceed,err);
  return 0;
}

static int CeedVectorSetArrayHost_OpenCL(const CeedVector vec,
                                       const CeedCopyMode cmode, CeedScalar *array) {
  int ierr;
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  switch (cmode) {
  case CEED_COPY_VALUES: ;
    CeedInt length;
    ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
    ierr = CeedMalloc(length, &data->h_array_allocated); CeedChk(ierr);
    data->h_array = data->h_array_allocated;

    if (array) memcpy(data->h_array, array, bytes(vec));
    break;
  case CEED_OWN_POINTER:
    data->h_array_allocated = array;
    data->h_array = array;
    break;
  case CEED_USE_POINTER:
    data->h_array = array;
    break;
  }
  data->memState = HOST_SYNC;
  return 0;
}

static int CeedVectorSetArrayDevice_OpenCL(const CeedVector vec,
    const CeedCopyMode cmode, cl_mem array) {
  int ierr; cl_int err;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void*)&ceed_data);

  switch (cmode) {
  case CEED_COPY_VALUES:
    data->d_array_allocated=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,bytes(vec),0,0);
    data->d_array=data->d_array_allocated;

    if (array) {
      err=clEnqueueWriteBuffer(ceed_data->queue,data->d_array,CL_TRUE,0,bytes(vec),array,0,0,0);
      CeedChk_OCL(ceed,err);
    }
    break;
  case CEED_OWN_POINTER:
    data->d_array_allocated = array;
    data->d_array = array;
    break;
  case CEED_USE_POINTER:
    data->d_array = array;
    break;
  }
  data->memState = DEVICE_SYNC;
  return 0;
}

// *****************************************************************************
// * Set the array used by a vector,
// * freeing any previously allocated array if applicable
// *****************************************************************************
static int CeedVectorSetArray_OpenCL(const CeedVector vec,
                                   const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   CeedScalar *array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);

  switch (mtype) {
  case CEED_MEM_HOST:
    ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
    return CeedVectorSetArrayHost_OpenCL(vec, cmode, array);
  case CEED_MEM_DEVICE:
    ierr = clReleaseMemObject(data->d_array_allocated); CeedChk_OCL(ceed, ierr);
    return CeedVectorSetArrayDevice_OpenCL(vec, cmode, (cl_mem) array);
  }
  return 1;
}

// *****************************************************************************
static int CeedHostSetValue(CeedScalar *h_array, CeedInt length,
                            CeedScalar val) {
  for (int i=0; i<length; i++) h_array[i] = val;
  return 0;
}

// *****************************************************************************
// * Set a vector to a value,
// *****************************************************************************
static int CeedVectorSetValue_OpenCL(CeedVector vec, CeedScalar val) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void*)&ceed_data);

  dbg("[CeedVectorSetValue][OpenCL]");

  int nparam=2;
  size_t size1=sizeof(cl_mem);
  size_t size2=sizeof(CeedScalar);
  void *args[] = {&nparam,&size1,(void *)&data->d_array,&size2,(void *)&val};

  switch(data->memState) {
  case HOST_SYNC:
    dbg("[CeedVectorSetValue][OpenCL][HOST_SYNC]");
    ierr = CeedHostSetValue(data->h_array, length, val);
    CeedChk(ierr);
    break;
  case NONE_SYNC:
    /*
      Handles the case where SetValue is used without SetArray.
      Default allocation then happens on the GPU.
    */
    dbg("[CeedVectorSetValue][OpenCL][NONE_SYNC]");
    if (data->d_array==NULL) {
      data->d_array_allocated=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,bytes(vec),0,0);
      CeedChk_OCL(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    data->memState = DEVICE_SYNC;
    ierr = run_kernel(ceed,data->setVector,data->setVector_work,args);
    CeedChk(ierr);
    break;
  case DEVICE_SYNC:
    dbg("[CeedVectorSetValue][OpenCL][DEVICE_SYNC]");
    ierr = run_kernel(ceed,data->setVector,data->setVector_work,args);
    CeedChk(ierr);
    break;
  case BOTH_SYNC:
    dbg("[CeedVectorSetValue][OpenCL][BOTH_SYNC]");
    ierr = CeedHostSetValue(data->h_array, length, val);
    CeedChk(ierr);
    ierr = run_kernel(ceed,data->setVector,data->setVector_work,args);
    CeedChk(ierr);
    break;
  }

  dbg("[CeedVectorSetValue][OpenCL]");

  return 0;
}


// *****************************************************************************
// * Get read-only access to a vector via the specified mtype memory type
// * on which to access the array. If the backend uses a different memory type,
// * this will perform a copy (possibly cached).
// *****************************************************************************
static int CeedVectorGetArrayRead_OpenCL(const CeedVector vec,
                                       const CeedMemType mtype,
                                       const CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void*)&ceed_data);

  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChk(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==DEVICE_SYNC) {
      ierr = CeedSyncD2H_OpenCL(vec);
      CeedChk(ierr);
      data->memState = BOTH_SYNC;
    }
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->d_array==NULL) {
      data->d_array_allocated=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,bytes(vec),0,0);
      CeedChk_OCL(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    if (data->memState==HOST_SYNC) {
      ierr = CeedSyncH2D_OpenCL(vec);
      CeedChk(ierr);
      data->memState = BOTH_SYNC;
    }
    *array = (CeedScalar *)data->d_array;
    break;
  }
  return 0;
}

// *****************************************************************************
static int CeedVectorGetArray_OpenCL(const CeedVector vec,
                                   const CeedMemType mtype,
                                   CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void*)&ceed_data);

  switch (mtype) {
  case CEED_MEM_HOST:
    if(data->h_array==NULL) {
      CeedInt length;
      ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
      ierr = CeedMalloc(length, &data->h_array_allocated);
      CeedChk(ierr);
      data->h_array = data->h_array_allocated;
    }
    if(data->memState==DEVICE_SYNC) {
      ierr = CeedSyncD2H_OpenCL(vec); CeedChk(ierr);
    }
    data->memState = HOST_SYNC;
    *array = data->h_array;
    break;
  case CEED_MEM_DEVICE:
    if (data->d_array==NULL) {
      data->d_array_allocated=clCreateBuffer(ceed_data->context,CL_MEM_READ_WRITE,bytes(vec),0,0);
      CeedChk_OCL(ceed, ierr);
      data->d_array = data->d_array_allocated;
    }
    if (data->memState==HOST_SYNC) {
      ierr = CeedSyncH2D_OpenCL(vec); CeedChk(ierr);
    }
    data->memState = DEVICE_SYNC;
    *array = (CeedScalar *) data->d_array;
    break;
  }
  return 0;
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_OpenCL(const CeedVector vec) {
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_OpenCL(const CeedVector vec) {
  return 0;
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_OpenCL(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void *)&data); CeedChk(ierr);
  ierr = clReleaseMemObject(data->d_array_allocated); CeedChk_OCL(ceed, ierr);
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_OpenCL(CeedInt n, CeedVector vec) {
  CeedVector_OpenCL *data;
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  dbg("[CeedVectorCreate][OpenCL]");

  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetValue",
                                CeedVectorSetValue_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArray",
                                CeedVectorGetArray_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "GetArrayRead",
                                CeedVectorGetArrayRead_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArray",
                                CeedVectorRestoreArray_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "RestoreArrayRead",
                                CeedVectorRestoreArrayRead_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "Destroy",
                                CeedVectorDestroy_OpenCL); CeedChk(ierr);

  ierr = CeedCalloc(1, &data); CeedChk(ierr);
  ierr = CeedVectorSetData(vec, (void *)&data); CeedChk(ierr);
  data->memState = NONE_SYNC;
  compile(ceed,data,"CeedVector",1,"length",n);

  dbg("[CeedVectorCreate][OpenCL]");
  return 0;
}
