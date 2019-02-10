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
#define CEED_DEBUG_COLOR 11
#include "ceed-opencl.h"
#include "ceed-backend.h"

// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedVector vec) {
  CeedInt length;
  CeedVectorGetLength(vec, &length);
  return length * sizeof(CeedScalar);
}

// *****************************************************************************
// * OpenCL data write/read functions
// *****************************************************************************
static inline void CeedWriteBuffer_OpenCL(const CeedVector vec) {
  Ceed_OpenCL *c;
  CeedVectorGetCeed(vec, (void*)&c);
  CeedVector_OpenCL *data;
  CeedVectorGetData(vec, (void*)&data);
  assert(c);
  assert(data);
  assert(data->h_array);
  clEnqueueWriteBuffer(c->queue, data->d_array, CL_TRUE,
                       0, bytes(vec), data->h_array, 0, NULL, NULL);
}
// *****************************************************************************
static inline void CeedReadBuffer_OpenCL(const CeedVector vec) {
  Ceed_OpenCL *c;
  CeedVectorGetCeed(vec, (void*)&c);
  CeedVector_OpenCL *data;
  CeedVectorGetData(vec, (void*)&data);
  assert(c);
  assert(data);
  assert(data->d_array);
  clEnqueueReadBuffer(c->queue, data->d_array, CL_TRUE,
                      0, bytes(vec), data->h_array, 0, NULL, NULL);
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
  CeedInt length;
  ierr = CeedVectorGetLength(vec, &length); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  dbg("[CeedVector][Set]");
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Only MemType = HOST supported");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  switch (cmode) {
  // Implementation will copy the values and not store the passed pointer.
  case CEED_COPY_VALUES:
    dbg("\t[CeedVector][Set] CEED_COPY_VALUES");
    ierr = CeedMalloc(vec->length, &data->h_array); CeedChk(ierr);
    data->h_array_allocated = data->h_array;
    if (array) memcpy(data->h_array, array, bytes(vec));
    if (array) CeedWriteBuffer_OpenCL(vec);
    break;
  // Implementation takes ownership of the pointer
  // and will free using CeedFree() when done using it
  case CEED_OWN_POINTER:
    dbg("\t[CeedVector][Set] CEED_OWN_POINTER");
    data->h_array = array;
    data->h_array_allocated = array;
    CeedWriteBuffer_OpenCL(vec);
    break;
  // Implementation can use and modify the data provided by the user
  case CEED_USE_POINTER:
    dbg("\t[CeedVector][Set] CEED_USE_POINTER");
    data->h_array = array;
    CeedWriteBuffer_OpenCL(vec);
    break;
  default: CeedError(ceed,1," OCCA backend no default error");
  }
  dbg("\t[CeedVector][Set] done");
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
  dbg("[CeedVector][Get]");
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  if (mtype != CEED_MEM_HOST)
    return CeedError(vec->ceed, 1, "Can only provide to HOST memory");
  if (!data->h_array) { // Allocate if array was not allocated yet
    dbg("[CeedVector][Get] Allocating");
    ierr = CeedVectorSetArray(vec, CEED_MEM_HOST, CEED_COPY_VALUES, NULL);
    CeedChk(ierr);
  }
  dbg("[CeedVector][Get] CeedSyncD2H_OpenCL");
  CeedReadBuffer_OpenCL(vec);
  *array = data->h_array;
  return 0;
}
// *****************************************************************************
static int CeedVectorGetArray_OpenCL(const CeedVector vec,
                                     const CeedMemType mtype,
                                     CeedScalar **array) {
  return CeedVectorGetArrayRead_OpenCL(vec,mtype,(const CeedScalar**)array);
}

// *****************************************************************************
// * Restore an array obtained using CeedVectorGetArray()
// *****************************************************************************
static int CeedVectorRestoreArrayRead_OpenCL(const CeedVector vec,
    const CeedScalar **array) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  dbg("[CeedVector][Restore]");
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  assert(data->h_array);
  assert(*array);
  CeedWriteBuffer_OpenCL(vec); // sync Host to Device
  *array = NULL;
  return 0;
}
// *****************************************************************************
static int CeedVectorRestoreArray_OpenCL(const CeedVector vec,
    CeedScalar **array) {
  return CeedVectorRestoreArrayRead_OpenCL(vec,(const CeedScalar**)array);
}

// *****************************************************************************
// * Destroy the vector
// *****************************************************************************
static int CeedVectorDestroy_OpenCL(const CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  CeedVector_OpenCL *data;
  ierr = CeedVectorGetData(vec, (void*)&data); CeedChk(ierr);
  dbg("[CeedVector][Destroy]");
  ierr = CeedFree(&data->h_array_allocated); CeedChk(ierr);
  clReleaseMemObject(data->d_array);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * Create a vector of the specified length (does not allocate memory)
// *****************************************************************************
int CeedVectorCreate_OpenCL(const CeedInt n, CeedVector vec) {
  int ierr;
  Ceed ceed;
  ierr = CeedVectorGetCeed(vec, &ceed); CeedChk(ierr);
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  CeedVector_OpenCL *data;
  dbg("[CeedVector][Create] n=%d", n);
  ierr = CeedSetBackendFunction(ceed, "Vector", vec, "SetArray",
                                CeedVectorSetArray_OpenCL); CeedChk(ierr);
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
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  data->d_array = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                 bytes(vec),
                                 NULL, NULL);
  ierr = CeedVectorSetData(vec, (void *)&data); CeedChk(ierr);
  return 0;
}
