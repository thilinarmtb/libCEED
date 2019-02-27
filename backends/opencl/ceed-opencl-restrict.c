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
  const CeedElemRestriction_OpenCL *data = r->data;
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
  const int identity = 1;
  // ***************************************************************************
  cl_int err;
  size_t globalSize = 1, localSize = 1;
  Ceed_OpenCL *ceed_data = ceed->data;

  if (identity) {
    dbg("[CeedElemRestriction][Apply] kRestrict[6]");
    CeedInt nelem_x_elemsize_x_ncomp = r->nelem*r->elemsize*r->ncomp;
    globalSize = (size_t) nelem_x_elemsize_x_ncomp;
    err = clSetKernelArg(data->kRestrict[6], 0, sizeof(CeedInt),
                          (void *)&nelem_x_elemsize_x_ncomp);
    err |= clSetKernelArg(data->kRestrict[6], 1, sizeof(cl_mem), (void *)&ud);
    err |= clSetKernelArg(data->kRestrict[6], 2, sizeof(cl_mem), (void *)&vd);

    dbg("[CeedElemRestriction][Apply] kRestrict[6] Setup done");
    localSize = 1;
    clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[6], 1, NULL,
                           &globalSize, &localSize, 0, NULL, NULL);
    dbg("[CeedElemRestriction][Apply] kRestrict[6] Setup done 1");
    clFinish(ceed_data->queue);
    dbg("[CeedElemRestriction][Apply] kRestrict[6] Setup done 2");
    dbg("[CeedElemRestriction][Apply] kRestrict[6] Setup done 3");
  } else if (restriction) {
    // Perform: v = r * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[0]");
      err  = clSetKernelArg(data->kRestrict[0], 0, sizeof(cl_mem), (void *)&id);
      size_t nelem_x_elemsize = r->nelem*r->elemsize;
      err |= clSetKernelArg(data->kRestrict[0], 1, sizeof(size_t),
                            (void *)&nelem_x_elemsize);
      err |= clSetKernelArg(data->kRestrict[0], 2, sizeof(cl_mem), (void *)&ud);
      err |= clSetKernelArg(data->kRestrict[0], 3, sizeof(cl_mem), (void *)&vd);

      localSize = 1;
      clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[0], 1, NULL,
                             &nelem_x_elemsize, &localSize, 0, NULL, NULL);
      clFlush(ceed_data->queue);
      clFinish(ceed_data->queue);
    } else {
      // v is (elemsize x ncomp x nelem), column-major
      if (ordering) {
        // u is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[1]");
        //occaKernelRun(data->kRestrict[1], occaInt(ncomp), id, ud, vd);
        err  = clSetKernelArg(data->kRestrict[1], 0, sizeof(CeedInt), &r->elemsize);
        err |= clSetKernelArg(data->kRestrict[1], 1, sizeof(cl_mem), &id);
        err |= clSetKernelArg(data->kRestrict[1], 2, sizeof(CeedInt), &r->ncomp);
        err |= clSetKernelArg(data->kRestrict[1], 3, sizeof(CeedInt), &r->ndof);
        err |= clSetKernelArg(data->kRestrict[1], 4, sizeof(CeedInt), &r->nelem);
        err |= clSetKernelArg(data->kRestrict[1], 5, sizeof(cl_mem), &ud);
        err |= clSetKernelArg(data->kRestrict[1], 6, sizeof(cl_mem), &vd);

        clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[1], 1, NULL,
                               &globalSize, &localSize, 0, NULL, NULL);
        clFinish(ceed_data->queue);
      } else {
        // u is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[2]");
        //occaKernelRun(data->kRestrict[2], occaInt(ncomp), id, ud, vd);
        err  = clSetKernelArg(data->kRestrict[2], 0, sizeof(CeedInt), &r->elemsize);
        err |= clSetKernelArg(data->kRestrict[2], 1, sizeof(cl_mem), &id);
        err |= clSetKernelArg(data->kRestrict[2], 2, sizeof(CeedInt), &r->ncomp);
        err |= clSetKernelArg(data->kRestrict[2], 3, sizeof(CeedInt), &r->nelem);
        err |= clSetKernelArg(data->kRestrict[2], 4, sizeof(cl_mem), &ud);
        err |= clSetKernelArg(data->kRestrict[2], 5, sizeof(cl_mem), &vd);

        clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[1], 1, NULL,
                               &globalSize, &localSize, 0, NULL, NULL);
        clFinish(ceed_data->queue);
      }
    }
  } else { // ******************************************************************
    // Note: in transpose mode, we perform: v += r^t * u
    if (ncomp == 1) {
      dbg("[CeedElemRestriction][Apply] kRestrict[3]");
      // occaKernelRun(occa->kRestrict[3], id, ud, vd);
      //occaKernelRun(data->kRestrict[6], tid, od, ud, vd);
      err  = clSetKernelArg(data->kRestrict[3], 0, sizeof(cl_mem), &tid);
      err |= clSetKernelArg(data->kRestrict[3], 1, sizeof(cl_mem), &od);
      err |= clSetKernelArg(data->kRestrict[3], 2, sizeof(cl_mem), &ud);
      err |= clSetKernelArg(data->kRestrict[3], 3, sizeof(cl_mem), &vd);

      clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[6], 1, NULL,
                             &globalSize, &localSize, 0, NULL, NULL);
      clFinish(ceed_data->queue);
    } else {
      // u is (elemsize x ncomp x nelem)
      if (ordering) {
        // v is (ndof x ncomp), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[7]");
        // occaKernelRun(data->kRestrict[4], occaInt(ncomp), id, ud, vd);
        //occaKernelRun(data->kRestrict[7], occaInt(ncomp), id, od,ud, vd);
        err  = clSetKernelArg(data->kRestrict[4], 0, sizeof(CeedInt), &ncomp);
        err |= clSetKernelArg(data->kRestrict[4], 1, sizeof(cl_mem), &id);
        err |= clSetKernelArg(data->kRestrict[4], 2, sizeof(cl_mem), &od);
        err |= clSetKernelArg(data->kRestrict[4], 3, sizeof(cl_mem), &ud);
        err |= clSetKernelArg(data->kRestrict[4], 4, sizeof(cl_mem), &vd);

        clEnqueueNDRangeKernel(ceed_data->queue, data->kRestrict[7], 1, NULL,
                               &globalSize, &localSize, 0, NULL, NULL);
        clFinish(ceed_data->queue);
      } else {
        // v is (ncomp x ndof), column-major
        dbg("[CeedElemRestriction][Apply] kRestrict[5]");
        // occaKernelRun(data->kRestrict[5], occaInt(ncomp), id, ud, vd);
        // occaKernelRun(data->kRestrict[8], occaInt(ncomp), id, od,ud, vd);
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
  const Ceed ceed = r->ceed;
  CeedElemRestriction_OpenCL *data = r->data;
  dbg("[CeedElemRestriction][Destroy]");
  cl_int err;
  for (int i=0; i<7; i++) {
    err = clReleaseKernel(data->kRestrict[i]);
    switch(err) {
      case CL_SUCCESS:
        printf("Successfully released kernel %d\n", i);
        break;
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
  clReleaseProgram(data->program);
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
                                    (1+r->ndof)*sizeof(CeedInt), NULL, NULL);
  data->d_tindices = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
                                    bytes(r), NULL, NULL);
  // ***************************************************************************
  ierr = CeedMalloc(ndof+1, &toffsets); CeedChk(ierr);
  ierr = CeedMalloc(elemsize*nelem, &tindices); CeedChk(ierr);
  if(indices) {
  CeedElemRestrictionOffset_OpenCL(r,indices,toffsets,tindices);
  //occaCopyPtrToMem(data->d_toffsets,toffsets,
  //                 (1+r->ndof)*sizeof(CeedInt),NO_OFFSET,NO_PROPS);
  clEnqueueWriteBuffer(ceed_data->queue, data->d_toffsets, CL_TRUE, 0,
                       (1+ndof)*sizeof(CeedInt), toffsets, 0, NULL, NULL);
  //occaCopyPtrToMem(data->d_tindices,tindices,bytes(r),NO_OFFSET,NO_PROPS);
  clEnqueueWriteBuffer(ceed_data->queue, data->d_tindices, CL_TRUE, 0,
                       bytes(r), tindices, 0, NULL, NULL);
  //occaCopyPtrToMem(data->d_indices,indices,bytes(r),NO_OFFSET,NO_PROPS);
  clEnqueueWriteBuffer(ceed_data->queue, data->d_indices, CL_TRUE, 0,
                       bytes(r), indices, 0, NULL, NULL);
  } else {
    data->identity = 1;
  }
  // ***************************************************************************
  dbg("[CeedElemRestriction][Create] Building kRestrict");

  dbg("[CeedElemRestriction][Create] Initialize kRestrict");

  // ***************************************************************************
  cl_int err;
  data->program = clCreateProgramWithSource(ceed_data->context, 1,
                  (const char **) &OpenCLKernels, NULL, &err);
  clBuildProgram(data->program, 1, &ceed_data->device_id, NULL, NULL, NULL);
  data->kRestrict[0] = clCreateKernel(data->program, "kRestrict0", &err);
  dbg("err after building kRestrict0: %d\n",err);
  data->kRestrict[1] = clCreateKernel(data->program, "kRestrict1", &err);
  dbg("err after building kRestrict1: %d\n",err);
  data->kRestrict[2] = clCreateKernel(data->program, "kRestrict2", &err);
  dbg("err after building kRestrict2: %d\n",err);
  data->kRestrict[3] = clCreateKernel(data->program, "kRestrict2", &err);
  dbg("err after building kRestrict3: %d\n",err);
  data->kRestrict[4] = clCreateKernel(data->program, "kRestrict2", &err);
  dbg("err after building kRestrict4: %d\n",err);
  data->kRestrict[5] = clCreateKernel(data->program, "kRestrict2", &err);
  dbg("err after building kRestrict5: %d\n",err);
  data->kRestrict[6] = clCreateKernel(data->program, "kRestrict6", &err);
  dbg("err after building kRestrict6: %d\n",err);
  // free local usage **********************************************************
  dbg("[CeedElemRestriction][Create] done");
  // free indices as needed ****************************************************
  if (cmode == CEED_OWN_POINTER) {
    ierr = CeedFree(&indices); CeedChk(ierr);
  }
  ierr = CeedFree(&toffsets); CeedChk(ierr);
  ierr = CeedFree(&tindices); CeedChk(ierr);
  ierr = CeedElemRestrictionSetData(r, (void*)&data); CeedChk(ierr);
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
