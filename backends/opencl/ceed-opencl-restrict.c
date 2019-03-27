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
#define CEED_DEBUG_COLOR 13
#include "ceed-opencl.h"
#include "ceed-backend.h"
// *****************************************************************************
// * Bytes used
// *****************************************************************************
static inline size_t bytes(const CeedElemRestriction res) {
  int ierr;
  CeedInt nelem, elemsize;
  ierr = CeedElemRestrictionGetNumElements(res, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(res, &elemsize); CeedChk(ierr);
  return nelem * elemsize * sizeof(CeedInt);
}

// *****************************************************************************
// * Restrict an L-vector to an E-vector or apply transpose
// *****************************************************************************
static
int CeedElemRestrictionApply_OpenCL(CeedElemRestriction r,
                                  CeedTransposeMode tmode,
                                  CeedTransposeMode lmode,
                                  CeedVector u, CeedVector v,
                                  CeedRequest *request) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedInt ncomp;
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  dbg("[CeedElemRestriction][Apply]");
  CeedElemRestriction_OpenCL *data;
  ierr = CeedElemRestrictionGetData(r, (void*)&data); CeedChk(ierr);
  const cl_mem id = data->d_indices;
  const cl_mem tid = data->d_tindices;
  const cl_mem od = data->d_toffsets;
  CeedVector_OpenCL *u_data;
  ierr = CeedVectorGetData(u, (void *)&u_data); CeedChk(ierr);
  CeedVector_OpenCL *v_data;
  ierr = CeedVectorGetData(v, (void *)&v_data); CeedChk(ierr);
  const cl_mem ud = u_data->d_array;
  const cl_mem vd = v_data->d_array;
  const CeedTransposeMode restriction = (tmode == CEED_NOTRANSPOSE);
  const CeedTransposeMode ordering = (lmode == CEED_NOTRANSPOSE);
  const bool identity = data->identity;
  // ***************************************************************************
  cl_int err;
  size_t globalSize, localSize = 1;
  Ceed_OpenCL *ceed_data = ceed->data;

  if (identity) {
    dbg("[CeedElemRestriction][Apply] kRestrict[6]");
    CeedInt nelem, elemsize;
    ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
    ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
    globalSize = (size_t) nelem*elemsize*ncomp;
    err |= clSetKernelArg(data->kRestrict[6], 0, sizeof(cl_mem), (void *)&ud);
    err |= clSetKernelArg(data->kRestrict[6], 1, sizeof(cl_mem), (void *)&vd);

    localSize = 1;
    clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[6], 1, NULL,
                           &globalSize, &localSize, 0, NULL, NULL);
    clFlush(ceed_data->queue);
    clFinish(ceed_data->queue);

    //Testing code - map to read, then unmap
    //cl_double* pointer = (cl_double*)clEnqueueMapBuffer(ceed_data->queue, vd, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_double), 0, NULL, NULL, NULL);
    //cl_double result = *pointer;
    //dbg("FIRST ELEMENT %g\n", result);
    //err = clEnqueueUnmapMemObject(ceed_data->queue, vd, pointer, NULL, NULL, NULL);
 
  } else if (restriction) {
    // Perform: v = r * u
    if (ncomp == 1) {
      //FIXME: Some of these arguments are constants now.
      dbg("[CeedElemRestriction][Apply] kRestrict[0] - Not implemented");
    } else {
      // v is (elemsize x ncomp x nelem), column-major
      if (ordering) {
        // u is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[1] - Not implemented");
      } else {
        // u is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[2] - Not implemented");
      }
    }
  } else { // ******************************************************************
    // Note: in transpose mode, we perform: v += r^t * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[3] - Not implemented ");
    } else {
      // u is (elemsize x ncomp x nelem)
      if (ordering) {
        // v is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[4] - Not implemented");
      } else {
        // v is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[5] - Not implemented");
      }
    }
  }
  if (request != CEED_REQUEST_IMMEDIATE && request != CEED_REQUEST_ORDERED)
    *request = NULL;
  dbg("[CeedElemRestriction][Apply] kRestrict[6] Done");
  return 0;
}

// *****************************************************************************
static int CeedElemRestrictionDestroy_OpenCL(CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedElemRestriction_OpenCL *data;
  ierr = CeedElemRestrictionGetData(r, (void*)&data); CeedChk(ierr);
  dbg("[CeedElemRestriction][Destroy]");
  cl_int err;
  for (int i=0; i<7; i++) {
    err = clReleaseKernel(data->kRestrict[i]);
    switch(err) {
    case CL_INVALID_KERNEL:
      printf("Invalid kernel %d\n", i);
      break;
    default:
      break;
    }
  }
  clReleaseMemObject(data->d_indices);
  clReleaseMemObject(data->d_toffsets);
  clReleaseMemObject(data->d_tindices);
  ierr = CeedFree(&data); CeedChk(ierr);
  dbg("[CeedElemRestriction][Destroy] Done.");
  return 0;
}

// *****************************************************************************
// * Compute the transposed Tindices and Toffsets from indices
// *****************************************************************************
static
int CeedElemRestrictionOffset_OpenCL(const CeedElemRestriction r,
                                   const CeedInt *indices,
                                   CeedInt *toffsets,
                                   CeedInt *tindices) {
  int ierr;
  CeedInt nelem, elemsize, ndof;
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  for (int i=0; i<=ndof; ++i) toffsets[i]=0;
  for (int e=0; e < nelem; ++e)
    for (int i=0; i < elemsize; ++i)
      ++toffsets[indices[elemsize*e+i]+1];
  for (int i = 1; i <= ndof; ++i)
    toffsets[i] += toffsets[i-1];
  for (int e = 0; e < nelem; ++e) {
    for (int i = 0; i < elemsize; ++i) {
      const int lid = elemsize*e+i;
      const int gid = indices[lid];
      tindices[toffsets[gid]++] = lid;
    }
  }
  for (int i = ndof; i > 0; --i)
    toffsets[i] = toffsets[i - 1];
  toffsets[0] = 0;
  return 0;
}

// *****************************************************************************
int CeedElemRestrictionCreate_OpenCL(const CeedMemType mtype,
                                   const CeedCopyMode cmode,
                                   const CeedInt *indices,
                                   const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  CeedInt ndof, nelem, ncomp, elemsize;
  ierr = CeedElemRestrictionGetNumDoF(r, &ndof); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumElements(r, &nelem); CeedChk(ierr);
  ierr = CeedElemRestrictionGetElementSize(r, &elemsize); CeedChk(ierr);
  ierr = CeedElemRestrictionGetNumComponents(r, &ncomp); CeedChk(ierr);
  dbg("[CeedElemRestriction][Create]");
  CeedElemRestriction_OpenCL *data;
  Ceed_OpenCL *ceed_data;
  ierr = CeedGetData(ceed, (void*)&ceed_data); CeedChk(ierr);
  //const bool ocl = ceed_data->ocl;
  //const occaDevice dev = ceed_data->device;
  // ***************************************************************************
  if (mtype != CEED_MEM_HOST)
    return CeedError(ceed, 1, "Only MemType = HOST supported");
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Apply",
                                CeedElemRestrictionApply_OpenCL); CeedChk(ierr);
  ierr = CeedSetBackendFunction(ceed, "ElemRestriction", r, "Destroy",
                                CeedElemRestrictionDestroy_OpenCL); CeedChk(ierr);
  // Allocating occa & device **************************************************
  dbg("[CeedElemRestriction][Create] Allocating");
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  // ***************************************************************************
  CeedInt *toffsets;
  CeedInt *tindices;
  data->d_indices  = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                    bytes(r), NULL, NULL);
  data->d_toffsets = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                    (1+ndof)*sizeof(CeedInt), NULL, NULL);
  data->d_tindices = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                    bytes(r), NULL, NULL);
  // ***************************************************************************
  ierr = CeedMalloc(ndof+1, &toffsets); CeedChk(ierr);
  ierr = CeedMalloc(elemsize*nelem, &tindices); CeedChk(ierr);
  if(indices) {
    CeedElemRestrictionOffset_OpenCL(r,indices,toffsets,tindices);
    clEnqueueWriteBuffer(ceed_data->queue, data->d_toffsets, CL_TRUE, 0,
                     (1+ndof)*sizeof(CeedInt), toffsets, 0, NULL, NULL);
    clEnqueueWriteBuffer(ceed_data->queue, data->d_tindices, CL_TRUE, 0,
                         bytes(r), tindices, 0, NULL, NULL);
    clEnqueueWriteBuffer(ceed_data->queue, data->d_indices, CL_TRUE, 0,
                         bytes(r), indices, 0, NULL, NULL);
  } else {
    data->identity = true;
  }
  // ***************************************************************************
  dbg("[CeedElemRestriction][Create] Building kRestrict");
  char *arch = ceed_data->arch;
  char constantDict[BUFSIZ];
  sprintf(constantDict, "{\"ndof\": %d,"
          "\"nelem\": %d,"
          "\"ncomp\": %d,"
          "\"elemsize\": %d,"
          "\"nelem_x_elemsize\": %d,"
          "\"nelem_x_elemsize_x_ncomp\": %d }",
          ndof, nelem, ncomp, elemsize, nelem*elemsize,nelem*elemsize*ncomp);

  ierr = CeedElemRestrictionSetData(r, (void*)&data); CeedChk(ierr);

  char *result;
  const char *pythonFile = "loopy_restrict.py";
  concat(&result, ceed_data->openclBackendDir, pythonFile);
  data->kRestrict[0] = createKernelFromPython("kRestrict0", arch, constantDict,
                       result, ceed);
  data->kRestrict[1] = createKernelFromPython("kRestrict1", arch, constantDict,
                       result, ceed);
  data->kRestrict[2] = createKernelFromPython("kRestrict2", arch, constantDict,
                       result, ceed);
  data->kRestrict[3] = createKernelFromPython("kRestrict2", arch, constantDict,
                       result, ceed);
  data->kRestrict[4] = createKernelFromPython("kRestrict2", arch, constantDict,
                       result, ceed);
  data->kRestrict[5] = createKernelFromPython("kRestrict2", arch, constantDict,
                       result, ceed);
  data->kRestrict[6] = createKernelFromPython("kRestrict6", arch, constantDict,
                       result, ceed);
  free(result);

  // free local usage **********************************************************
  dbg("[CeedElemRestriction][Create] done");
  // free indices as needed ****************************************************
  if (cmode == CEED_OWN_POINTER) {
    ierr = CeedFree(&indices); CeedChk(ierr);
  }
  ierr = CeedFree(&toffsets); CeedChk(ierr);
  ierr = CeedFree(&tindices); CeedChk(ierr);
  return 0;
}
// *****************************************************************************
int CeedElemRestrictionCreateBlocked_OpenCL(const CeedMemType mtype,
    const CeedCopyMode cmode,
    const CeedInt *indices,
    const CeedElemRestriction r) {
  int ierr;
  Ceed ceed;
  ierr = CeedElemRestrictionGetCeed(r, &ceed); CeedChk(ierr);
  return CeedError(ceed, 1, "Backend does not implement blocked restrictions");
}
