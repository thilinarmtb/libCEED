import numpy as np
import loopy as lp
import sys

from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

# setup
#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

#Idea: Have function take platform id and device id and have it figure out workgroup sizes itself

def gen_set_array_base(version=0, constants={}, fp_format=np.float64, target=lp.OpenCLTarget()):
    kernel_data = [
        lp.GlobalArg("array", fp_format),
        lp.ValueArg("val", fp_format)
    ]
    if not "length" in constants:
        kernel_data += [ lp.ValueArg("length", np.int32) ]

    k = lp.make_kernel(
        "{ [i]: 0<=i<length }",
        """
        array[i] = val
        """,
        name="kSetValue",
        kernel_data=kernel_data,
        assumptions="length > 0",
        target=target
    )
 
    k = lp.fix_parameters(k, **constants)
    
    return k

def transform_set_array(kernel, **kwargs):

    if "local_size" in kwargs:
        local_size = kwargs["local_size"]
    else:
        local_size = 1
         
    if "work_items" in kwargs:
        work_items = kwargs["length"]
    else:
        work_items = 1    

    slabs = (0,0) if work_items % local_size == 0 else (0,1)
    kernel = lp.split_iname(kernel, "i", inner_length=local_size, 
        outer_tag="g.0", inner_tag="l.0", slabs=slabs)
    return kernel
   

def gen_set_array(version=0, constants={}, fp_format=np.float64, \
        target=lp.OpenCLTarget()):
    k = gen_set_array_base(version=version, constants=constants, fp_format=fp_format, target=target)
    if "length" in constants:
        k = transform_set_array(k,work_items=constants["length"], local_size=128)    
    else:
        k = transform_set_array(k,local_size=128)
    return k

'''
def gen_set_array(constants={}, version=0, arch="INTEL_CPU", fp_format=np.float64, \
        target=lp.OpenCLTarget()):

    kernel_data = [
        lp.GlobalArg("array", fp_format),
        lp.ValueArg("val", fp_format)
    ]
    if constants=={}:
        kernel_data += [
            lp.ValueArg("length", np.int32), 
        ]
    k = lp.make_kernel(
        "{ [i]: 0<=i<length }",
        """
        array[i] = val
        """,
        name="kSetValue",
        kernel_data=kernel_data,
        assumptions="length > 0",
        target=target
    )
 
    k = lp.fix_parameters(k, **constants)

    if arch == "AMD_GPU":
        local_size = 64
    elif arch == "NVIDIA_GPU":
        local_size = 32
    else:
        local_size = 128

    local_size = opt_params["local_size"] if "local_size" in opt_params else 1

    if "length" in constants:
        work_items = constants["length"]
        local_size = min(local_size, work_items)
        global_size = int(np.ceil(work_items/local_size))*local_size
    else:
        work_items = 1    

    slabs = (0,0) if work_items % local_size == 0 else (0,1)
    k = lp.split_iname(k, "i", inner_length=local_size, 
        outer_tag="g.0", inner_tag="l.0", slabs=slabs)

    return k
'''

kern = gen_set_array()
print(kern)
code = lp.generate_code_v2(kern).device_code()
print(code)
