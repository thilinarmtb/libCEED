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
#define CEED_DEBUG_COLOR 249
#include "ceed-opencl.h"

// *****************************************************************************
// * buildKernel
// *****************************************************************************
static int CeedBasisBuildKernel(CeedBasis basis) {
  const Ceed ceed = basis->ceed;
  const Ceed_OpenCL *ceed_data = ceed->data;
  CeedBasis_OpenCL *data = basis->data;
  // ***************************************************************************
  const int dim = basis->dim;
  const int P1d = basis->P1d;
  const int Q1d = basis->Q1d;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = ncomp*CeedIntPow(Q1d,dim);
  const CeedInt vsize = ncomp*CeedIntPow(P1d,dim);
  // ***************************************************************************
  const CeedElemRestriction er = data->er; assert(er);
  const CeedInt nelem = er->nelem;
  const CeedInt elemsize = er->elemsize;
  const bool ocl = ceed_data->ocl;
  // ***************************************************************************
  char compileOptions[BUFSIZ], tmp[BUFSIZ];
  sprintf(tmp, "-Ddim=%d", dim);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-DP1d=%d", P1d);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-DQ1d=%d", Q1d);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-Dnc=%d", ncomp);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-Dncomp=%d", ncomp);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-Dnqpt=%d", nqpt);
  strcat(compileOptions, tmp);
  sprintf(tmp, ",-Dvsize=%d", vsize);
  strcat(compileOptions, tmp);

  dbg("[CeedBasis][BK] compileOptions=%s", compileOptions);
  dbg("[CeedBasis][BK] dim=%d",dim);
  dbg("[CeedBasis][BK] P1d=%d",P1d);
  dbg("[CeedBasis][BK] Q1d=%d",Q1d);
  dbg("[CeedBasis][BK] ncomp=%d",ncomp);
  dbg("[CeedBasis][BK] nqpt=%d",nqpt);
  dbg("[CeedBasis][BK] vsize=%d",vsize);
  // ***************************************************************************
  sprintf(tmp, "-Dnelem=%d", nelem);
  strcat(compileOptions, tmp);
  sprintf(tmp,",-Delemsize=%d", elemsize);
  strcat(compileOptions, tmp);

  dbg("[CeedBasis][BK] nelem=%d",nelem);
  dbg("[CeedBasis][BK] elemsize=%d",elemsize);
  // ***************************************************************************
  // OpenCL check for this requirement
  const CeedInt elem_tile_size = (nelem>OPENCL_TILE_SIZE)?OPENCL_TILE_SIZE:nelem;
  // OCCA+MacOS implementation needs that for now (if DeviceID targets a CPU)
  const CeedInt tile_size = ocl?1:elem_tile_size;
  sprintf(tmp, ",-DTILE_SIZE=%d", tile_size);
  strcat(compileOptions, tmp);

  dbg("[CeedBasis][BK] TILE_SIZE=%d",tile_size);
  // ***************************************************************************
  const CeedInt M1d = (Q1d>P1d)?Q1d:P1d;
  sprintf(tmp, ",-DM1d=%d", M1d);
  strcat(compileOptions, tmp);
  const CeedInt MPow = CeedIntPow(M1d,dim-1);
  const CeedInt tmpSz = ncomp*M1d*CeedIntPow(M1d,dim-1);
  sprintf(tmp,",-DtmpSz=%d", tmpSz);
  strcat(compileOptions, tmp);

  dbg("[CeedBasis][BK] nelem=%d, ncomp=%d, M1d=%d, MPow=%d",
      nelem,ncomp,M1d,MPow);
  dbg("[CeedBasis][BK] dim=%d, ncomp=%d, P1d=%d, Q1d=%d, M1d=%d ",
      dim,ncomp,P1d,Q1d,M1d);
  const CeedInt elems_x_tmpSz = nelem*tmpSz;
  dbg("[CeedBasis][BK] elems_x_tmpSz=%d",elems_x_tmpSz);

  data->tmp0 = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
		elems_x_tmpSz*sizeof(CeedScalar),NULL,NULL);
  data->tmp1 = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
	  	elems_x_tmpSz*sizeof(CeedScalar),NULL,NULL);
  // ***************************************************************************
  cl_int err;
  data->program = clCreateProgramWithSource(ceed_data->context, 1, OpenCLKernels, NULL, &err);
  clBuildProgram(data->program, 1, &ceed_data->device_id, compileOptions, NULL, NULL);
  data->kZero   = clCreateKernel(data->program, "kZero"  , &err);
  data->kInterp = clCreateKernel(data->program, "kInterp", &err);
  data->kGrad   = clCreateKernel(data->program, "kGrad"  , &err);
  data->kWeight = clCreateKernel(data->program, "kWeight", &err);
  // free local usage **********************************************************
  return 0;
}

// *****************************************************************************
// * TENSORS: Contracts on the middle index
// *          NOTRANSPOSE: V_ajc = T_jb U_abc
// *          TRANSPOSE:   V_ajc = T_bj U_abc
// * CeedScalars are used here, not CeedVectors: we don't touch it yet
// *****************************************************************************
static int CeedTensorContract_OpenCL(CeedInt A, CeedInt B, CeedInt C, CeedInt J,
                                   const CeedScalar *t, CeedTransposeMode tmode,
                                   const CeedInt Add,
                                   const CeedScalar *u, CeedScalar *v) {
  const CeedInt transpose = tmode == CEED_TRANSPOSE;
  const CeedInt tstride0 = transpose?1:B;
  const CeedInt tstride1 = transpose?J:1;
  for (CeedInt a=0; a<A; a++) {
    for (CeedInt j=0; j<J; j++) {
      if (!Add)
        for (CeedInt c=0; c<C; c++)
          v[(a*J+j)*C+c] = 0.0;
      for (CeedInt b=0; b<B; b++) {
        for (CeedInt c=0; c<C; c++) {
          v[(a*J+j)*C+c] += t[j*tstride0 + b*tstride1] * u[(a*B+b)*C+c];
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisApplyElems_OpenCL
// *****************************************************************************
int CeedBasisApplyElems_OpenCL(CeedBasis basis, CeedInt QnD,
                             CeedTransposeMode tmode, CeedEvalMode emode,
                             const CeedVector u, CeedVector v) {
  const Ceed ceed = basis->ceed;
  const Ceed_OpenCL *ceed_data = ceed->data;
  CeedBasis_OpenCL *data = basis->data;
  const CeedInt ready =  data->ready;
  // ***************************************************************************
  // We were waiting for the CeedElemRestriction to fill nelem and elemsize
  if (!ready) {
    data->ready=true;
    CeedBasisBuildKernel(basis);
  }
  // ***************************************************************************
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);
  // ***************************************************************************
  cl_int err;

  size_t globalSize, localSize;
  // Number of work items in each local work group
  localSize = 64;
  // ***************************************************************************
  if (transpose) {
    dbg("[CeedBasis][ApplyElems] transpose");
    const CeedVector_OpenCL *v_data = v->data;
    const cl_mem d_v = v_data->d_array;
    err  = clSetKernelArg(data->kZero, 0, sizeof(cl_mem), &d_v);
    clEnqueueNDRangeKernel(ceed_data->queue, data->kZero, 1, NULL,
		    &globalSize, &localSize, 0, NULL, NULL);
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    dbg("[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_INTERP");
    const cl_mem d_tmp0 = data->tmp0;
    const cl_mem d_tmp1 = data->tmp1;
    const cl_mem d_interp1d = data->interp1d;
    const CeedVector_OpenCL *u_data = u->data; assert(u_data);
    const CeedVector_OpenCL *v_data = v->data; assert(v_data);
    const cl_mem d_u = u_data->d_array;
    const cl_mem d_v = v_data->d_array;

    //occaKernelRun(data->kInterp,occaInt(QnD),
    //              occaInt(transpose),occaInt(tmode),
    //              d_tmp0, d_tmp1, d_interp1d,
    //              d_u, d_v);
    err  = clSetKernelArg(data->kInterp, 0, sizeof(CeedInt), &QnD);
    err |= clSetKernelArg(data->kInterp, 1, sizeof(CeedInt), &transpose);
    err |= clSetKernelArg(data->kInterp, 2, sizeof(CeedInt), &tmode);
    err |= clSetKernelArg(data->kInterp, 3, sizeof(cl_mem), &d_tmp0);
    err |= clSetKernelArg(data->kInterp, 4, sizeof(cl_mem), &d_tmp1);
    err |= clSetKernelArg(data->kInterp, 5, sizeof(cl_mem), &d_interp1d);
    err |= clSetKernelArg(data->kInterp, 6, sizeof(cl_mem), &d_u);
    err |= clSetKernelArg(data->kInterp, 7, sizeof(cl_mem), &d_v);

    clEnqueueNDRangeKernel(ceed_data->queue, data->kInterp, 1, NULL,
		    &globalSize, &localSize, 0, NULL, NULL);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_GRAD");
    const cl_mem d_tmp0 = data->tmp0;
    const cl_mem d_tmp1 = data->tmp1;
    const cl_mem d_grad1d = data->grad1d;
    const cl_mem d_interp1d = data->interp1d;
    const CeedVector_OpenCL *u_data = u->data; assert(u_data);
    const CeedVector_OpenCL *v_data = v->data; assert(v_data);
    const cl_mem d_u = u_data->d_array;
    const cl_mem d_v = v_data->d_array;
    //occaKernelRun(data->kGrad,occaInt(QnD),
    //              occaInt(transpose),occaInt(tmode),
    //              d_tmp0,d_tmp1,d_grad1d,d_interp1d,
    //              d_u, d_v);
    err  = clSetKernelArg(data->kGrad, 0, sizeof(CeedInt), &QnD);
    err |= clSetKernelArg(data->kGrad, 1, sizeof(CeedInt), &transpose);
    err |= clSetKernelArg(data->kGrad, 2, sizeof(CeedInt), &tmode);
    err |= clSetKernelArg(data->kGrad, 3, sizeof(cl_mem), &d_tmp0);
    err |= clSetKernelArg(data->kGrad, 4, sizeof(cl_mem), &d_tmp1);
    err |= clSetKernelArg(data->kGrad, 5, sizeof(cl_mem), &d_grad1d);
    err |= clSetKernelArg(data->kGrad, 6, sizeof(cl_mem), &d_interp1d);
    err |= clSetKernelArg(data->kGrad, 7, sizeof(cl_mem), &d_u);
    err |= clSetKernelArg(data->kGrad, 8, sizeof(cl_mem), &d_v);

    clEnqueueNDRangeKernel(ceed_data->queue, data->kGrad, 1, NULL,
		    &globalSize, &localSize, 0, NULL, NULL);
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    dbg("[CeedBasis][ApplyElems] CEED_EVAL_WEIGHT");
    if (transpose)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    const CeedInt Q1d = basis->Q1d;
    const cl_mem d_qw = data->qweight1d;
    const CeedVector_OpenCL *v_data = v->data; assert(v_data);
    const cl_mem d_v = v_data->d_array;

    err  = clSetKernelArg(data->kWeight, 0, sizeof(CeedInt), &QnD);
    err |= clSetKernelArg(data->kWeight, 1, sizeof(CeedInt), &Q1d);
    err |= clSetKernelArg(data->kWeight, 2, sizeof(cl_mem), &d_qw);
    err |= clSetKernelArg(data->kWeight, 3, sizeof(cl_mem), &d_v);

    //occaKernelRun(data->kWeight,occaInt(QnD),occaInt(Q1d),d_qw,d_v);
    clEnqueueNDRangeKernel(ceed_data->queue, data->kWeight, 1, NULL,
		    &globalSize, &localSize, 0, NULL, NULL);
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisApply_OpenCL
// *****************************************************************************
static int CeedBasisApply_OpenCL(CeedBasis basis, CeedInt nelem,
                               CeedTransposeMode tmode, CeedEvalMode emode,
                               const CeedScalar *u, CeedScalar *v) {
  int ierr;
  const CeedInt dim = basis->dim;
  const CeedInt ncomp = basis->ncomp;
  const CeedInt nqpt = ncomp*CeedIntPow(basis->Q1d, dim);
  const CeedInt transpose = (tmode == CEED_TRANSPOSE);

  if (nelem != 1)
    return CeedError(basis->ceed, 1,
                     "This backend does not support BasisApply for multiple elements");
  // ***************************************************************************
  if (transpose) {
    const CeedInt vsize = ncomp*CeedIntPow(basis->P1d, dim);
    //dbg("[CeedBasis][Apply] transpose");
    for (CeedInt i = 0; i < vsize; i++)
      v[i] = 0.0;
  }
  // ***************************************************************************
  if (emode == CEED_EVAL_NONE) {
    //dbg("[CeedBasis][Apply] CEED_EVAL_NONE");
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_INTERP) {
    const CeedInt P = transpose?basis->Q1d:basis->P1d;
    const CeedInt Q = transpose?basis->P1d:basis->Q1d;
    CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
    //dbg("[CeedBasis][Apply] CEED_EVAL_INTERP");
    CeedScalar tmp[2][ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    for (CeedInt d=0; d<dim; d++) {
      ierr = CeedTensorContract_OpenCL(pre, P, post, Q,
                                     basis->interp1d,
                                     tmode, transpose&&(d==dim-1),
                                     d==0?u:tmp[d%2],
                                     d==dim-1?v:tmp[(d+1)%2]);
      CeedChk(ierr);
      pre /= P;
      post *= Q;
    }
    if (!transpose) v += nqpt;
    else u += nqpt;
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_GRAD) {
    const CeedInt P = transpose?basis->Q1d:basis->P1d;
    const CeedInt Q = transpose?basis->P1d:basis->Q1d;
    //dbg("[CeedBasis][Apply] CEED_EVAL_GRAD, P=%d, Q=%d",P,Q);
    CeedScalar tmp[2][ncomp*Q*CeedIntPow(P>Q?P:Q, dim-1)];
    for (CeedInt p=0; p<dim; p++) {
      CeedInt pre = ncomp*CeedIntPow(P, dim-1), post = 1;
      for (CeedInt d=0; d<dim; d++) {
        ierr = CeedTensorContract_OpenCL(pre, P, post, Q,
                                       (p==d)?basis->grad1d:basis->interp1d,
                                       tmode, transpose&&(d==dim-1),
                                       d==0?u:tmp[d%2], d==dim-1?v:tmp[(d+1)%2]);
        CeedChk(ierr);
        pre /= P;
        post *= Q;
      }
      if (!transpose) v += nqpt;
      else u += nqpt;
    }
  }
  // ***************************************************************************
  if (emode & CEED_EVAL_WEIGHT) {
    //dbg("[CeedBasis][Apply] CEED_EVAL_WEIGHT");
    if (transpose)
      return CeedError(basis->ceed, 1,
                       "CEED_EVAL_WEIGHT incompatible with CEED_TRANSPOSE");
    // *************************************************************************
    CeedInt Q = basis->Q1d;
    for (CeedInt d=0; d<dim; d++) {
      const CeedInt pre = CeedIntPow(Q, dim-d-1), post = CeedIntPow(Q, d);
      for (CeedInt i=0; i<pre; i++) {
        for (CeedInt j=0; j<Q; j++) {
          for (CeedInt k=0; k<post; k++) {
            v[(i*Q + j)*post + k] =
              basis->qweight1d[j] * (d == 0 ? 1 : v[(i*Q + j)*post + k]);
          }
        }
      }
    }
  }
  return 0;
}

// *****************************************************************************
// * CeedBasisDestroy_OpenCL
// *****************************************************************************
static int CeedBasisDestroy_OpenCL(CeedBasis basis) {
  int ierr;
  const Ceed ceed = basis->ceed;
  CeedBasis_OpenCL *data = basis->data;
  dbg("[CeedBasis][Destroy]");
  clReleaseKernel(data->kZero);
  clReleaseKernel(data->kInterp);
  clReleaseKernel(data->kGrad);
  clReleaseKernel(data->kWeight);
  clReleaseMemObject(data->qref1d);
  clReleaseMemObject(data->qweight1d);
  clReleaseMemObject(data->interp1d);
  clReleaseMemObject(data->grad1d);
  ierr = CeedFree(&data); CeedChk(ierr);
  return 0;
}

// *****************************************************************************
// * CeedBasisCreateTensorH1_OpenCL
// *****************************************************************************
int CeedBasisCreateTensorH1_OpenCL(CeedInt dim, CeedInt P1d, CeedInt Q1d,
                                 const CeedScalar *interp1d,
                                 const CeedScalar *grad1d,
                                 const CeedScalar *qref1d,
                                 const CeedScalar *qweight1d,
                                 CeedBasis basis) {
  int ierr;
  CeedBasis_OpenCL *data;
  Ceed ceed = basis->ceed;
  const Ceed_OpenCL *ceed_data = ceed->data;
  dbg("[CeedBasis][CreateTensorH1]");
  // ***************************************************************************
  ierr = CeedCalloc(1,&data); CeedChk(ierr);
  basis->data = data;
  // ***************************************************************************
  assert(qref1d);
  //data->qref1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  //occaCopyPtrToMem(data->qref1d,qref1d,Q1d*sizeof(CeedScalar),NO_OFFSET,NO_PROPS);
  data->qref1d = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
		  Q1d*sizeof(CeedScalar), NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, data->qref1d, CL_TRUE, 0,
		  Q1d*sizeof(CeedScalar), qref1d, 0, NULL, NULL);
  // ***************************************************************************
  assert(qweight1d);
  //data->qweight1d = occaDeviceMalloc(dev,Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  //occaCopyPtrToMem(data->qweight1d,qweight1d,Q1d*sizeof(CeedScalar),NO_OFFSET,
  //                 NO_PROPS);
  data->qweight1d = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
		  Q1d*sizeof(CeedScalar), NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, data->qweight1d, CL_TRUE, 0,
		  Q1d*sizeof(CeedScalar), qweight1d, 0, NULL, NULL);
  // ***************************************************************************
  assert(interp1d);
  //data->interp1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  //occaCopyPtrToMem(data->interp1d,interp1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,
  //                 NO_PROPS);
  data->interp1d = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
		  P1d*Q1d*sizeof(CeedScalar), NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, data->interp1d, CL_TRUE, 0,
		  P1d*Q1d*sizeof(CeedScalar), interp1d, 0, NULL, NULL);
  // ***************************************************************************
  assert(grad1d);
  //data->grad1d = occaDeviceMalloc(dev,P1d*Q1d*sizeof(CeedScalar),NULL,NO_PROPS);
  //occaCopyPtrToMem(data->grad1d,grad1d,P1d*Q1d*sizeof(CeedScalar),NO_OFFSET,
  //                 NO_PROPS);
  data->grad1d = clCreateBuffer(ceed_data->context, CL_MEM_READ_WRITE,
		  P1d*Q1d*sizeof(CeedScalar), NULL, NULL);
  clEnqueueWriteBuffer(ceed_data->queue, data->grad1d, CL_TRUE, 0,
		  P1d*Q1d*sizeof(CeedScalar), grad1d, 0, NULL, NULL);
  // ***************************************************************************
  basis->Apply = CeedBasisApply_OpenCL;
  basis->Destroy = CeedBasisDestroy_OpenCL;
  return 0;
}

// *****************************************************************************
// * CeedBasisCreateH1_OpenCL
// *****************************************************************************
int CeedBasisCreateH1_OpenCL(CeedElemTopology topo, CeedInt dim,
                          CeedInt ndof, CeedInt nqpts,
                          const CeedScalar *interp,
                          const CeedScalar *grad,
                          const CeedScalar *qref,
                          const CeedScalar *qweight,
                          CeedBasis basis) {
  return CeedError(basis->ceed, 1, "Backend does not implement non-tensor bases");
}