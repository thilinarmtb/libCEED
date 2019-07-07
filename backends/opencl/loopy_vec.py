import numpy as np
import loopy as lp
import sys

# setup
#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

#Idea: Have function take platform id and device id and have it figure out workgroup sizes itself
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
        name="setVector",
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

    global_size = -1
    if "length" in constants:
        work_items = constants["length"]
        local_size = min(local_size, work_items)
        global_size = int(np.ceil(work_items/local_size))*local_size
    else:
        work_items = 1    

    slabs = (0,0) if work_items % local_size == 0 else (0,1)
    k = lp.split_iname(k, "i", inner_length=local_size, 
        outer_tag="g.0", inner_tag="l.0", slabs=slabs)

    return lp.generate_code_v2(k).device_code()
