// *****************************************************************************
#ifndef _ceed_opencl_h
#define _ceed_opencl_h

#include <ceed-impl.h>
#include <ceed-backend.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define OPENCL_TILE_SIZE 32
#define MAX_BUF 100000

// *****************************************************************************
// * CeedVector_OpenCL struct
// *****************************************************************************
typedef struct {
  CeedScalar *h_array;
  CeedScalar *h_array_allocated;
  cl_mem d_array;
} CeedVector_OpenCL;

// *****************************************************************************
// * CeedElemRestriction_OpenCL struct
// *****************************************************************************
#define CEED_OPENCL_NUM_RESTRICTION_KERNEL 9
typedef struct {
  cl_mem d_indices;
  cl_mem d_toffsets;
  cl_mem d_tindices;
  cl_kernel kRestrict[CEED_OPENCL_NUM_RESTRICTION_KERNEL];
  cl_kernel kRestrict6;
  char *compleOptions;
  bool identity;
} CeedElemRestriction_OpenCL;

// *****************************************************************************
// * CeedBasis_OpenCL struct
// *****************************************************************************
typedef struct {
  _Bool ready;
  CeedElemRestriction er;
  cl_mem qref1d;
  cl_mem qweight1d;
  cl_mem interp1d;
  cl_mem grad1d;
  cl_mem tmp0,tmp1;
  cl_kernel kZero,kInterp,kGrad,kWeight;
} CeedBasis_OpenCL;

// *****************************************************************************
// * CeedOperator_OpenCL struct
// *****************************************************************************
typedef struct {
  CeedVector *Evecs; /// E-vectors needed to apply operator (in followed by out)
  CeedScalar **Edata;
  CeedVector *evecsin;   /// Input E-vectors needed to apply operator
  CeedVector *evecsout;   /// Output E-vectors needed to apply operator
  CeedVector *qvecsin;   /// Input Q-vectors needed to apply operator
  CeedVector *qvecsout;   /// Output Q-vectors needed to apply operator
  CeedInt    numein;
  CeedInt    numeout;
} CeedOperator_OpenCL;

// *****************************************************************************
// * CeedQFunction_OpenCL struct
// *****************************************************************************
#define N_MAX_IDX 16
typedef struct {
  _Bool ready;
  CeedInt idx,odx;
  CeedInt iOf7[N_MAX_IDX];
  CeedInt oOf7[N_MAX_IDX];
  int nc, dim, nelem, elemsize, e;
  double epsilon;
  cl_mem o_indata, o_outdata;
  cl_mem d_ctx, d_idx, d_odx;
  char *qFunctionName;
  char *pythonFile;
  cl_kernel kQFunctionApply;
  char *compleOptions;
  CeedOperator op;
} CeedQFunction_OpenCL;

// *****************************************************************************
// * Ceed_OpenCL struct
// *****************************************************************************
typedef struct {
  bool debug;
  bool ocl;
  char *arch;
  char *openclBackendDir;
  cl_platform_id cpPlatform;     // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  bool gpu;
  bool cpu;
} Ceed_OpenCL;

// *****************************************************************************
// CEED_DEBUG_COLOR default value, forward CeedDebug* declarations & dbgcl macros
// *****************************************************************************
#ifndef CEED_DEBUG_COLOR
#define CEED_DEBUG_COLOR 0
#endif
void CeedDebugImpl_OpenCL(const Ceed,const char*,...);
void CeedDebugImpl256_OpenCL(const Ceed,const unsigned char,const char*,...);
#define CeedDebug(ceed,format, ...) CeedDebugImpl_OpenCL(ceed,format, ## __VA_ARGS__)
#define CeedDebug256(ceed,color, ...) CeedDebugImpl256_OpenCL(ceed,color, ## __VA_ARGS__)
#define dbg(...) CeedDebug256(ceed,(unsigned char)CEED_DEBUG_COLOR, ## __VA_ARGS__)

// *****************************************************************************
CEED_INTERN int CeedBasisCreateTensorH1_OpenCL(CeedInt dim,
    CeedInt P1d, CeedInt Q1d, const CeedScalar *interp1d, const CeedScalar *grad1d,
    const CeedScalar *qref1d, const CeedScalar *qweight1d, CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisCreateH1_OpenCL(CeedElemTopology topo,
    CeedInt dim, CeedInt ndof, CeedInt nqpts,
    const CeedScalar *interp1d,
    const CeedScalar *grad1d,
    const CeedScalar *qref1d,
    const CeedScalar *qweight1d,
    CeedBasis basis);

// *****************************************************************************
CEED_INTERN int CeedBasisApplyElems_OpenCL(CeedBasis basis, CeedInt Q,
    CeedTransposeMode tmode, CeedEvalMode emode, const CeedVector u, CeedVector v);

// *****************************************************************************
CEED_INTERN int CeedOperatorCreate_OpenCL(CeedOperator op);

// *****************************************************************************
CEED_INTERN int CeedQFunctionCreate_OpenCL(CeedQFunction qf);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreate_OpenCL(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedElemRestrictionCreateBlocked_OpenCL(const CeedMemType mtype,
    const CeedCopyMode cmode, const CeedInt *indices,
    const CeedElemRestriction res);

// *****************************************************************************
CEED_INTERN int CeedVectorCreate_OpenCL(CeedInt n, CeedVector vec);

// *****************************************************************************
CEED_INTERN cl_kernel createKernelFromPython(char *kernelName, char *arch,
    char *constantDict, char *pythonFile, Ceed ceed);

CEED_INTERN void concat(char **result, const char *s1, const char *s2);
#endif
