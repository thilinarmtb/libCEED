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

#VERSION = LMODE | TMODE | INDICES
LMODE = 4 
TMODE = 2
INDICES = 1

#Idea: Have function take platform id and device id and have it figure out workgroup sizes itself
def get_restrict(constants={}, version=0, arch="INTEL_CPU", fp_format=np.float64, \
        target=lp.OpenCLTarget()):

    kernel_data = [
        lp.GlobalArg("u", fp_format),
        lp.GlobalArg("v", fp_format, for_atomic=True if version >= 4 else False),
        lp.GlobalArg("indices", np.int32)
    ]
    if constants=={}:
        kernel_data += [
            lp.ValueArg("elemsize", np.int32), 
            lp.ValueArg("ncomp", np.int32),
            lp.ValueArg("ndof", np.int32),
            lp.ValueArg("esize", np.int32)]
    else:
        constants["esize"] = constants["nelem"]*constants["elemsize"]*constants["ncomp"]

    loopyCode = """
                e := i / (ncomp * elemsize) 
                d := (i / elemsize) % ncomp
                s := i % elemsize"""

    suffix = ""    
    if version == int(not LMODE) | int(not TMODE) | int(not INDICES):
        suffix = """ 
                 v[i] = u[s + elemsize*e + ndof*d] 
                 """
    elif version == int(not LMODE) | int(not TMODE) | INDICES:
        suffix = """
                 v[i] = u[indices[s + elemsize*e] + ndof*d]
                 """        
    elif version == int(not LMODE) | TMODE | int(not INDICES):
        suffix = """
                 v[i] = u[ncomp*(s + elemsize*e) + d]
                 """        
    elif version == int(not LMODE) | TMODE | INDICES:
        suffix = """
                 v[i] = u[ncomp * indices[s + elemsize*e] + d]
                 """        
    elif version == LMODE | int(not TMODE) | int(not INDICES):
        suffix = """
                 v[s + elemsize*e + ndof*d] = v[s + elemsize*e + ndof*d] + u[i] {atomic}
                 """        
    elif version == LMODE | int(not TMODE) | INDICES:
        suffix = """
                 v[indices[s + elemsize*e] + ndof*d] = v[indices[s + elemsize*e] + ndof*d] + u[i] {atomic}
                 """        
    elif version == LMODE | TMODE | int(not INDICES):
        suffix = """
                 v[ncomp*(s + elemsize*e) + d] = v[ncomp*(s + elemsize*e) + d] + u[i] {atomic}
                 """        
    elif version == LMODE | TMODE | INDICES:
        suffix = """
                 v[ncomp * indices[s + elemsize*e] + d] = v[ncomp * indices[s + elemsize*e] + d] + u[i] {atomic}
                 """      
    else:
        raise Exception("Invalid version value in generate_kRestrict()")  

    

    loopyCode += suffix
    k = lp.make_kernel(
        "{ [i]: 0<=i<esize }",
        loopyCode,
        name="kRestrict",
        kernel_data=kernel_data,
        assumptions="esize > 0",
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
    if "esize" in constants:
        work_items = constants["esize"]
        local_size = min(local_size, work_items)
        global_size = int(np.ceil(work_items/local_size))*local_size
    else:
        work_items = 1    

    slabs = (0,0) if work_items % local_size == 0 else (0,1)
    k = lp.split_iname(k, "i", inner_length=local_size, 
        outer_tag="g.0", inner_tag="l.0", slabs=slabs)

    return lp.generate_code_v2(k).device_code()
