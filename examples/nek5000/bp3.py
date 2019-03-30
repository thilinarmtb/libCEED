import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def generate_diffsetupf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    dtypes = {
        "ctx": fp_format,
        "Q": np.int32,
        "iOf7": np.int32,
        "oOf7": np.int32,
        "in": fp_format,
        "out": fp_format
    }
    kernel_data = ["ctx", "Q", "iOf7", "oOf7", "in", "out"]
 

    diffsetupf = lp.make_kernel(
        ["{ [i,j,k]: 0<=i<Q }"],
        """
        # For reference
        #    *x = in + iOf7[0],
        #    *J = in + iOf7[1];
        #    *w = in + iOf7[2];
        #    *rho = out + oOf7[0]
        #    *rhs = out + oOf7[1]

        if 0
            iOf7[0] = 0
            oOf7[0] = 0
            ctx[0] = 0
        end

        M_PI := 3.14159265358979323846
        iind0(a) := i + iOf7[0] + a*Q
        iind1 := i + iOf7[1]
        iind2 := i + iOf7[2]
        oind1 := i + oOf7[1]
        oind0(a) := oOf7[0] + i + Q*a
    
        J(jj,kk) := in[3*Q*(kk-1) + Q*(jj-1) + iind1]
       
        # Could precompute these, but compilers are
        # typically good at optimizing common subexpressions 
        A11 := J(2,2)*J(3,3) - J(2,3)*J(3,2)
        A12 := J(1,3)*J(3,2) - J(1,2)*J(3,3)
        A13 := J(1,2)*J(2,3) - J(1,3)*J(2,2)
        A21 := J(2,3)*J(3,1) - J(2,1)*J(3,3)
        A22 := J(1,1)*J(3,3) - J(1,3)*J(3,1)
        A23 := J(1,3)*J(2,1) - J(1,1)*J(2,3)
        A31 := J(2,1)*J(3,2) - J(2,2)*J(3,1)
        A32 := J(1,2)*J(3,1) - J(1,1)*J(3,2)
        A33 := J(1,1)*J(2,2) - J(1,2)*J(2,1)

        s := (J(1,1)*A11 + J(2,1)*A12 + J(3,1)*A13)
        w := in[iind2] / s
        rho := in[iind2] * s

        out[oind0(0)] = w * (A11*A11 + A12*A12 + A13*A13)
        out[oind0(1)] = w * (A11*A21 + A12*A22 + A13*A23)
        out[oind0(2)] = w * (A11*A31 + A12*A32 + A13*A33)
        out[oind0(3)] = w * (A21*A21 + A22*A22 + A23*A23)
        out[oind0(4)] = w * (A21*A31 + A22*A32 + A23*A33)
        out[oind0(5)] = w * (A31*A31 + A32*A32 + A33*A33)

        out[oind1] = rho * M_PI**2 * (1**2 + 2**2 + 3**2)  * \
                            sin(M_PI*(0 + 1*in[iind0(0)])) * \
                            sin(M_PI*(1 + 2*in[iind0(1)])) * \
                            sin(M_PI*(2 + 3*in[iind0(2)])) 
        """,
        name="diffsetupf",
        assumptions="Q > 0",
        kernel_data=kernel_data,
        target=target,
        )

    diffsetupf = lp.fix_parameters(diffsetupf, **constants)
    #diffsetupf = lp.tag_inames(diffsetupf, [("j", "unr"), ("k", "unr")])

    diffsetupf = lp.add_and_infer_dtypes(diffsetupf, dtypes)
    
    return diffsetupf

def generate_diffusionf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    dtypes = {
        "ctx": fp_format,
        "Q": np.int32,
        "iOf7": np.int32,
        "oOf7": np.int32,
        "in": fp_format,
        "out": fp_format
    }
    kernel_data = ["ctx", "Q", "iOf7", "oOf7", "in", "out"]
    
    diffusionf = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        #    For Reference
        #    *u = in  + iOf7[0],
        #    *rho = in  + iOf7[1];
        #    *v = out + oOf7[0];

        ug(a) := in[iOf7[0] + i + Q*a]
        rhog(a) := in[iOf7[1] + i + Q*a]
        oind0(a) := oOf7[0] + i + Q*a

        if 0
            iOf7[0] = 0
            ctx[0]  = 0
            oOf7[0] = 0
        end

        out[oind0(0)] = rhog(0)*ug(0) + rhog(1)*ug(1) + rhog(2)*ug(2)
        out[oind0(1)] = rhog(1)*ug(0) + rhog(3)*ug(1) + rhog(4)*ug(2)  
        out[oind0(2)] = rhog(2)*ug(0) + rhog(4)*ug(1) + rhog(5)*ug(2)
        """,
        name="diffusionf",
        kernel_data=kernel_data,
        assumptions="Q > 0",
        target=target
        )

    diffusionf = lp.add_and_infer_dtypes(diffusionf, dtypes)
 
    return diffusionf

kernel_name = sys.argv[1]
arch = sys.argv[2]
constants = json.loads(sys.argv[3])
 
if kernel_name == 'diffusionf':
    k = generate_diffusionf(constants, arch)
elif kernel_name == 'diffsetupf':
    k = generate_diffsetupf(constants, arch)
else:
    print("Invalid kernel name: {}".format(kernel_name))
    sys.exit(1)
 
code = lp.generate_code_v2(k).device_code()
try:
    print(code)
except IOError:
    print('An IO error occured.')
except:
    print('An unknown error occured.')

'''
diffsetupf = generate_diffsetupf()
code = lp.generate_code_v2(diffsetupf).device_code()
print(diffsetupf)
print()
print(code)
print()

diffusionf = generate_diffusionf()
code = lp.generate_code_v2(diffusionf).device_code()
print(diffsetupf)
print()
print(code)
print()
'''
