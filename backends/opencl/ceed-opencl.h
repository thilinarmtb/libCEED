// *****************************************************************************
// This header should be included at the top before any other standard headers.
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

int OpenCL_test_00(void);
