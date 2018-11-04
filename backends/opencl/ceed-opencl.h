// *****************************************************************************
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
  clmem d_array;
} CeedVector_OpenCL;

// *****************************************************************************
// * CeedElemRestriction_OpenCL struct
// *****************************************************************************
#define CEED_OCCA_NUM_RESTRICTION_KERNEL 9
typedef struct {
  clmem d_indices;
  clmem d_toffests;
  clmem d_tindices;
  cl_kernel kRestrict[CEED_OCCA_NUM_RESTRICTION_KERNEL];
}







// *****************************************************************************
int OpenCL_test_00(void);
