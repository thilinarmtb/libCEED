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

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;
using namespace py::literals;

int run_kernel(Ceed ceed,
    cl_kernel kernel,
    CeedWork_OpenCL *work,
    void **args) {
  return 0;
}

int compile(Ceed ceed, void *data,
    const char *type,
    int nparams, ...) {
  py::scoped_interpreter guard{};
  py::object get_restrict = py::module::import("loopy_restrict").attr("get_restirct");
  auto constants = py::dict();

  va_list args;
  va_start(args, nparams);

  const char *param_name;
  int param_value;
  for(int i=0;i<nparams;i++) {
    param_name=va_arg(args,const char *);
    param_value=va_arg(args,int);
    constants[param_name] = param_value;
  }

  if(strcmp(type,"CeedRestrict")==0){
    CeedElemRestriction_OpenCL *data_ = (CeedElemRestriction_OpenCL *) data;
    int indices = data_->h_ind ? 1 : 0;
    int lmode[2] = {0,4};
    int tmode[2] = {0,2};
    cl_kernel kernels[2][2] = {data_->noTrNoTr,data_->noTrTr,data_->trNoTr,data_->trTr};

    for(int ll=0;ll<2;ll++) {
      for(int tt=0;tt<2;tt++) {
        int version=lmode[ll]|tmode[tt]|indices;
        py::object temp = get_restrict("constants"_a=constants,"version"_a=version);
      }
    }

    printf("CeedRestrict kernels are not implemented yet\n");
    exit(1);
  } else if(strcmp(type,"CeedBasis")==0){
    CeedBasis_OpenCL *data_ = (CeedBasis_OpenCL *) data;

    printf("CeedBasis kernels are not implemented yet\n");
    exit(1);
  } else if(strcmp(type,"setVector")==0){
  }

  return 0;
}
