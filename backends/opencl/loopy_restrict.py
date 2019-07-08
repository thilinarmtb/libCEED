import numpy as np
import loopy as lp
import sys

#from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

# setup
#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

#VERSION = TMODE | LMODE | INDICES
LMODE = 2 
TMODE = 4
INDICES = 1

#Idea: Have function take platform id and device id and have it figure out workgroup sizes itself
def get_restrict(version=0, constants={}, arch="INTEL_CPU", fp_format=np.float64, \
        target=lp.OpenCLTarget()):

    kernel_data = [
        lp.GlobalArg("u", fp_format),
        lp.GlobalArg("v", fp_format, for_atomic=True if version >= 4 else False)
    ]

    if version & 1 == 1:
        kernel_data.append(lp.GlobalArg("indices", np.int32))       

    if not "elemsize" in constants:
        kernel_data.append(lp.ValueArg("elemsize", np.int32))
    if not "ncomp" in constants:
        kernel_data.append(lp.ValueArg("ncomp", np.int32))
    if not "ndof" in constants:
        kernel_data.append(lp.ValueArg("ndof", np.int32))
    if not "nelem" in constants and not "elemsize" in constants and not "ncomp" in constants:
        kernel_data.append(lp.ValueArg("esize", np.int32))
    else:
        constants["esize"] = constants["nelem"]*constants["elemsize"]*constants["ncomp"]
    '''
    if constants=={}:
        kernel_data += [
            lp.ValueArg("elemsize", np.int32), 
            lp.ValueArg("ncomp", np.int32),
            lp.ValueArg("ndof", np.int32),
            lp.ValueArg("esize", np.int32)]
    else:
        constants["esize"] = constants["nelem"]*constants["elemsize"]*constants["ncomp"]
    '''

    loopyCode = """
                <int> e = (i / (ncomp * elemsize)) 
                d := ((i / elemsize) % ncomp)
                s := (i % elemsize)
                """

    '''
    versionList = [
                    "v[i] = u[(s) + elemsize*(e) + ndof*(d)]",
                    "v[i] = u[indices[(s) + elemsize*(e)] + ndof*(d)]",
                    "v[i] = u[ncomp*((s) + elemsize*(e)) + (d)]",
                    "v[i] = u[ncomp * indices[(s) + elemsize*(e)] + (d)]",
                    "v[(s) + elemsize*(e) + ndof*(d)] = v[(s) + elemsize*(e) + ndof*(d)] + u[i] {atomic}",
                    "v[indices[(s) + elemsize*(e)] + ndof*(d)] = v[indices[(s) + elemsize*(e)] + ndof*(d)] + u[i] {atomic}",
                    "v[ncomp*((s) + elemsize*(e)) + (d)] = v[ncomp*((s) + elemsize*(e)) + (d)] + u[i] {atomic}",
                    "v[ncomp * indices[(s) + elemsize*(e)] + (d)] = v[ncomp * indices[(s) + elemsize*(e)] + (d)] + u[i] {atomic}" ]
    '''

    versionList = [
                    "v[i] = u[s + elemsize*e + ndof*d]",
                    "v[i] = u[indices[s + elemsize*e] + ndof*d]",
                    "v[i] = u[ncomp*(s + elemsize*e) + d]",
                    "v[i] = u[ncomp * indices[s + elemsize*e] + d]",
                    "v[s + elemsize*e + ndof*d] = v[s + elemsize*e + ndof*d] + u[i] {atomic}",
                    "v[indices[s + elemsize*e] + ndof*d] = v[indices[s + elemsize*e] + ndof*d] + u[i] {atomic}",
                    "v[ncomp*(s + elemsize*e) + d] = v[ncomp*(s + elemsize*e) + d] + u[i] {atomic}",
                    "v[ncomp * indices[s + elemsize*e] + d] = v[ncomp * indices[s + elemsize*e] + d] + u[i] {atomic}" ]
    #'''

    loopyCode += versionList[version] + "\n"
    '''    
    if version == int(not LMODE) | int(not TMODE) | int(not INDICES):
        loopyCode += """ 
                     v[i] = u[s + elemsize*e + ndof*d] 
                     """
    elif version == int(not LMODE) | int(not TMODE) | INDICES:
        loopyCode += """
                     v[i] = u[indices[s + elemsize*e] + ndof*d]
                     """        
    elif version == int(not LMODE) | TMODE | int(not INDICES):
        loopyCode += """
                     v[i] = u[ncomp*(s + elemsize*e) + d]
                     """        
    elif version == int(not LMODE) | TMODE | INDICES:
        loopyCode += """
                     v[i] = u[ncomp * indices[s + elemsize*e] + d]
                     """        
    elif version == LMODE | int(not TMODE) | int(not INDICES):
        loopyCode += """
                     v[s + elemsize*e + ndof*d] = v[s + elemsize*e + ndof*d] + u[i] {atomic}
                     """        
    elif version == LMODE | int(not TMODE) | INDICES:
        loopyCode += """
                     v[indices[s + elemsize*e] + ndof*d] = v[indices[s + elemsize*e] + ndof*d] + u[i] {atomic}
                     """        
    elif version == LMODE | TMODE | int(not INDICES):
        loopyCode += """
                     v[ncomp*(s + elemsize*e) + d] = v[ncomp*(s + elemsize*e) + d] + u[i] {atomic}
                     """        
    elif version == LMODE | TMODE | INDICES:
        loopyCode += """
                     v[ncomp * indices[s + elemsize*e] + d] = v[ncomp * indices[s + elemsize*e] + d] + u[i] {atomic}
                     """      
    else:
        raise Exception("Invalid version value in generate_kRestrict()")  
    '''
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

'''
constants = {"nelem": 3, "ncomp": 1, "ndof": 4, "elemsize": 2} 
for v in range(8):
    print(str(v) + "=======================")
    code = get_restrict(version=v,constants=constants)
    print(code)
'''
