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

# Currently works only for 3D
def generate_masssetupf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    masssetupf = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        D := 3
        v(a, b) := in[(a*D + b)*Q + i + iOf7[1]]

        if false
            oOf7[0] = 0
            iOf7[1] = 0
            ctx[0] = 0
        end

        det := in[iOf7[2] + i] * (
                   v(0,0) * (v(1,1)*v(2,2) - v(1,2)*v(2,1))
                 - v(0,1) * (v(1,0)*v(2,2) - v(1,2)*v(2,0))
                 + v(0,2) * (v(1,0)*v(2,1) - v(1,1)*v(2,0)))
        sum := v(0,0)**2 + v(0,1)**2 + v(0,2)**2

        out[oOf7[0] + i] = det
        out[oOf7[1] + i] = det * sqrt(sum) 

        #out[oOf7[0] + i] = in[iOf7[0]+i]
        #out[oOf7[1] + i] = in[iOf7[1]+i]    
        #in[i] = 111
        #out[i] = 999.0
        """,
        name="masssetupf",
        assumptions="Q > 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=target
        )

    masssetupf = lp.add_and_infer_dtypes(masssetupf, {
        "ctx": fp_format, "in": fp_format, "out": fp_format,
        "oOf7": np.int32,"iOf7": np.int32 
        })
    
    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        workgroup_size = 128

    #massfsetupf = lp.split_iname(masssetupf, "i", workgroup_size, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
 
    return masssetupf


def generate_massf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    massf = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        o_os := i + oOf7[0]
        i_os0 := i + iOf7[0]
        i_os1 := i + iOf7[1]
        if false
            ctx[0] = 0 # Compiler will hopefully remove
        end

        out[o_os] = in[i_os0] * in[i_os1]
        """,
        name="massf",
        assumptions="Q > 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=target
        )

    #massf = lp.fix_parameters(massf, constants)

    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        workgroup_size = 128

    #massf = lp.split_iname(massf, "i", workgroup_size, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    
    massf = lp.add_and_infer_dtypes(massf, {
        "ctx": fp_format,
        "in": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32 
    })

    return massf

kernel_name = sys.argv[1]
arch = sys.argv[2]
constants = json.loads(sys.argv[3])
 
if kernel_name == 'massf':
    k = generate_massf(constants, arch)
elif kernel_name == 'masssetupf':
    k = generate_masssetupf(constants, arch)
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
