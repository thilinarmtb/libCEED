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

    dtypes = {
        "ctx": fp_format, 
        "in": fp_format, 
        "out": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32 
    }

    kernel_data = ["ctx","iOf7", "oOf7", "in", "out"]
    if constants == {}:
        kernel_data += ["ctx", "Q","iOf7", "oOf7", "in", "out"]

    k = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        # Force loopy to make these variables pointers
        # even if they are not used or commented out
        if false
            <> dummy = oOf7[0]
            dummy = iOf7[1]
            dummy = ctx[0]
        end

        D := 3
        v(a, b, c) := in[(a*D + b)*Q + i + iOf7[c]]

        det := in[iOf7[2] + i] * (
                   v(0,0,1) * (v(1,1,1)*v(2,2,1) - v(1,2,1)*v(2,1,1))
                 - v(0,1,1) * (v(1,0,1)*v(2,2,1) - v(1,2,1)*v(2,0,1))
                 + v(0,2,1) * (v(1,0,1)*v(2,1,1) - v(1,1,1)*v(2,0,1)))
        sum := v(0,0,0)**2 + v(0,1,0)**2 + v(0,2,0)**2

        out[oOf7[0] + i] = det
        out[oOf7[1] + i] = det * sqrt(sum) 
        """,
        name="masssetupf",
        assumptions="Q > 0",
        kernel_data=kernel_data,
        target=target
    )

    k = lp.fix_parameters(k, **constants)
    k = lp.add_and_infer_dtypes(k, dtypes)
    
    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        workgroup_size = 128

    global_size = -1
    if "Q" in constants:
        global_size = constants["Q"]
        workgroup_size = min(workgroup_size, global_size)

    slabs = (0,0) if global_size % workgroup_size == 0 else (0,1) 
    k = lp.split_iname(k, "i", workgroup_size,
            outer_tag="g.0", inner_tag="l.0", slabs=slabs)
 
    code = lp.generate_code_v2(k).device_code()
 
    outDict = {
        "kernel": code,
        "work_dim": 1,
        "local_work_size": [workgroup_size]
    }
    if global_size > 0:
       outDict.update({"global_work_size": [global_size]}),

    return outDict

def generate_massf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    dtypes = {
        "ctx": fp_format, 
        "in": fp_format, 
        "out": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32 
    }
    
    kernel_data = ["ctx","iOf7", "oOf7", "in", "out"]
    if constants == {}:
        kernel_data += ["ctx", "Q","iOf7", "oOf7", "in", "out"]

    k = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        o_os  := i + oOf7[0]
        i_os0 := i + iOf7[0]
        i_os1 := i + iOf7[1]

        # Force loopy to make these variables pointers
        # even if they are not used or commented out
        if false
            <> dummy = oOf7[0]
            dummy = iOf7[0]
            dummy = ctx[0]
        end

        out[o_os] = in[i_os0] * in[i_os1]
        """,
        name="massf",
        assumptions="Q > 0",
        kernel_data=kernel_data,
        target=target
        )

    k = lp.fix_parameters(k, **constants)
    k = lp.add_and_infer_dtypes(k, dtypes)
    
    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        workgroup_size = 128

    global_size = -1
    if "Q" in constants:
        global_size = constants["Q"]
        workgroup_size = min(workgroup_size, global_size)

    slabs = (0,0) if global_size % workgroup_size == 0 else (0,1) 
    k = lp.split_iname(k, "i", workgroup_size,
            outer_tag="g.0", inner_tag="l.0", slabs=slabs)
   
    code = lp.generate_code_v2(k).device_code()
    print(k)
 
    outDict = {
        "kernel": code,
        "work_dim": 1,
        "local_work_size": [workgroup_size]
    }
    if global_size > 0:
       outDict.update({"global_work_size": [global_size]}),

    return outDict

'''
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
'''

#code = generate_masssetupf(constants={"Q": 277})["kernel"]
#print(code)
#print()
