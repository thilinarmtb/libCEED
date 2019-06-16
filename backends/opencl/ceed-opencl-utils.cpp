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

#ifdef __cplusplus
extern "C" {
#endif

#define restrict __restrict__

#include <ceed-impl.h>
#include "ceed-opencl.h"
#include "unistd.h"

#ifdef __cplusplus
}
#endif

#include <cstring>
#include <cstdarg>

CEED_INTERN int run_kernel(Ceed ceed,
    cl_kernel kernel,
    CeedWork_OpenCL *work,
    void **args) {
  return 0;
}

CEED_INTERN int compile(Ceed ceed, void *data,
    const char *type,
    int nparams, ...) {
  const char *param_name;
  int param_value;

  va_list args;
  va_start(args, nparams);

  if(strcmp(type,"CeedRestrict")==0) {
    CeedElemRestriction_OpenCL *data_ = (CeedElemRestriction_OpenCL *) data;
  } else if(strcmp(type,"CeedBasis")==0) {
    CeedBasis_OpenCL *data_ = (CeedBasis_OpenCL *) data;
  }

  return 0;
}
