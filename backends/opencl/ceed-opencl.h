// *****************************************************************************
#include <ceed-impl.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// *****************************************************************************
// * CeedVector_OpenCL struct
// *****************************************************************************
typedef struct {
  CeedScalar *h_array;
  cl_mem d_array;
} CeedVector_OpenCL;

// *****************************************************************************
// * CeedElemRestriction_OpenCL struct
// *****************************************************************************
#define CEED_OPENCL_NUM_RESTRICTION_KERNEL 9
typedef struct {
  cl_mem d_indices;
  cl_mem d_toffests;
  cl_mem d_tindices;
  cl_kernel kRestrict[CEED_OPENCL_NUM_RESTRICTION_KERNEL];
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
  CeedScalar **qdata;
  CeedScalar **qdata_alloc; /// Inputs followed by outputs
  CeedScalar **indata;
  CeedScalar **outdata;
  CeedInt    numein;
  CeedInt    numeout;
  CeedInt    numqin;
  CeedInt    numqout;
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
  cl_mem o_indata, o_outdata;
  cl_mem d_ctx, d_idx, d_odx;
  char *oklPath;
  const char *qFunctionName;
  cl_kernel kQFunctionApply;
  CeedOperator op;
} CeedQFunction_OpenCL;

// *****************************************************************************
// * Ceed_OpenCL struct
// *****************************************************************************
typedef struct {
  _Bool debug;
  _Bool ocl;
  char *libceed_dir;
  cl_platform_id cpPlatform;        // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  cl_program program;               // program
} Ceed_OpenCL;

// *****************************************************************************
CEED_INTERN int CeedVectorCreate_OpenCL(CeedInt n, CeedVector vec);
