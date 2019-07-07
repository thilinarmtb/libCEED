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

#include <iostream>
#include <string>

#include <cstring>
#include <cstdarg>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
namespace py = pybind11;
using namespace py::literals;

cl_kernel createKernelFromSource(Ceed ceed,
    const char* kernelCode,
    const char *kernelName) {
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void **)&ceed_data);

  cl_int err;
  cl_program program;
  program = clCreateProgramWithSource(ceed_data->context,1,(const char **)&kernelCode,
      NULL, &err);
  switch(err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_CONTEXT:
    fprintf(stderr, "OpenCL backend: Invalid context.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  }

  err = clBuildProgram(program, 1, &ceed_data->device_id, 
    "-cl-fast-relaxed-math -cl-denorms-are-zero", NULL, NULL);

  // Determine the size of the log
  size_t log_size;
  char *log;
  switch(err) {
  case CL_SUCCESS:
    break;
  case CL_INVALID_PROGRAM:
    fprintf(stderr, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_INVALID_DEVICE:
    fprintf(stderr, "OpenCL backend: Invalid device.");
    break;
  case CL_INVALID_BINARY:
    fprintf(stderr, "OpenCL backend: Invalid binary.");
    break;
  case CL_INVALID_BUILD_OPTIONS:
    fprintf(stderr, "OpenCL backend: Invalid build options.");
    break;
  case CL_INVALID_OPERATION:
    fprintf(stderr, "OpenCL backend: Invalid operation.");
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    fprintf(stderr, "OpenCL backend: Compiler not available.");
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    clGetProgramBuildInfo(program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          0, NULL,
                          &log_size);
    // Allocate memory for the log
    log = (char *) malloc(log_size);
    // Get the log
    clGetProgramBuildInfo(program, ceed_data->device_id, CL_PROGRAM_BUILD_LOG,
                          log_size,
                          log, NULL);
    // Print the log
    //printf("%s\n", log);
    fprintf(stderr, "OpenCL backend: Build program failure.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  }

  cl_kernel kernel   = clCreateKernel(program,kernelName,&err);
  switch(err) {
  case CL_INVALID_PROGRAM:
    fprintf(stderr, "OpenCL backend: Invalid program.");
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    fprintf(stderr, "OpenCL backend: Invalid program executable.");
    break;
  case CL_INVALID_KERNEL_NAME:
    fprintf(stderr, "OpenCL backend: Invalid kernel name.");
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    fprintf(stderr, "OpenCL backend: Invalid kernel definition.");
    break;
  case CL_INVALID_VALUE:
    fprintf(stderr, "OpenCL backend: Invalid value.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  return kernel;
}

int compile(Ceed ceed, void *data,
    const char *type,
    int nparams, ...) {
  va_list args;
  va_start(args, nparams);

  const char *param_name;
  int param_value;
  auto constants = py::dict();
  for(int i=0;i<nparams;i++) {
    param_name=va_arg(args,const char *);
    param_value=va_arg(args,int);
    constants[param_name] = param_value;
  }

  if(strcmp(type,"CeedRestrict")==0){
    py::object get_restrict = py::module::import("loopy_restrict").attr("get_restirct");
    CeedElemRestriction_OpenCL *rstrct = (CeedElemRestriction_OpenCL *) data;
    int indices = rstrct->h_ind ? 1 : 0;
    int lmode[2] = {0,4};
    int tmode[2] = {0,2};
    cl_kernel kernels[2][2] = {rstrct->noTrNoTr,rstrct->noTrTr,rstrct->trNoTr,rstrct->trTr};

    for(int ll=0;ll<2;ll++) {
      for(int tt=0;tt<2;tt++) {
        int version=lmode[ll]|tmode[tt]|indices;
        py::object temp = get_restrict("constants"_a=constants,"version"_a=version);
      }
    }

    printf("CeedRestrict kernels are not implemented yet\n");
    exit(1);
  } else if(strcmp(type,"CeedBasis")==0){
    py::object get_basis = py::module::import("loopy_basis").attr("get_basis");
    CeedBasis_OpenCL *basis = (CeedBasis_OpenCL *) data;

    printf("CeedBasis kernels are not implemented yet\n");
    exit(1);
  } else if(strcmp(type,"CeedVector")==0){
    py::object gen_set_array = py::module::import("loopy_vec").attr("gen_set_array");
    CeedVector_OpenCL *vector = (CeedVector_OpenCL *) data;
    py::object kernel = gen_set_array("constants"_a=constants);
    std::string source = py::cast<std::string>(kernel);
    vector->setVector=createKernelFromSource(ceed,source.c_str(),"setVector");
  }
  return 0;
}

int run_kernel(Ceed ceed,
    cl_kernel kernel,
    CeedWork_OpenCL *work,
    void **args) {
  Ceed_OpenCL *ceed_data;
  CeedGetData(ceed,(void **)&ceed_data);

  cl_int err;
  int nparam=*((int *)args[0]);
  for(int i=0;i<nparam;i++){
    size_t size=*((size_t*)args[2*i+1]);
    void *ptr=args[2*i+2];
    err|=clSetKernelArg(kernel,i,size,ptr);
  }

  err = clEnqueueNDRangeKernel(ceed_data->queue,kernel,work->work_dim,NULL,
      &work->global_work_size,&work->local_work_size,0,NULL,NULL);

  switch(err) {
  case CL_INVALID_PROGRAM_EXECUTABLE:
    fprintf(stderr, "OpenCL backend: Invalid program executable.");
    break;
  case CL_INVALID_COMMAND_QUEUE:
    fprintf(stderr, "OpenCL backend: Invalid command queue.");
    break;
  case CL_INVALID_KERNEL:
    fprintf(stderr, "OpenCL backend: Invalid kernel.");
    break;
  case CL_INVALID_CONTEXT:
    fprintf(stderr, "OpenCL backend: Invalid context.");
    break;
  case CL_INVALID_KERNEL_ARGS:
    fprintf(stderr, "OpenCL backend: Invalid kernel args.");
    break;
  case CL_INVALID_WORK_DIMENSION:
    fprintf(stderr, "OpenCL backend: Invalid work dimension.");
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    fprintf(stderr, "OpenCL backend: Invalid work group size.");
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    fprintf(stderr, "OpenCL backend: Invalid work item size.");
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    fprintf(stderr, "OpenCL backend: Invalid global offset.");
    break;
  case CL_OUT_OF_RESOURCES:
    fprintf(stderr, "OpenCL backend: Out of host resources.");
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    fprintf(stderr, "OpenCL backend: Mem object allocation failure.");
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    fprintf(stderr, "OpenCL backend: Invalid event wait list.");
    break;
  case CL_OUT_OF_HOST_MEMORY:
    fprintf(stderr, "OpenCL backend: Out of host memory.");
    break;
  default:
    break;
  }

  clFlush(ceed_data->queue);
  clFinish(ceed_data->queue);
  return 0;
}

int init_loopy() {
  py::initialize_interpreter();
  return 0;
}
