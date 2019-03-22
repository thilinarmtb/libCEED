import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def generate_kRestrict0(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kRestrict0 = lp.make_kernel(
        "{ [i]: 0<=i<nelem_x_elemsize}",
        """
        vv[i] = uu[indices[i]]
        """,
        name="kRestrict0",
        assumptions="nelem_x_elemsize > 0",
        target=target 
        )

    kRestrict0 = lp.fix_parameters(kRestrict0, **constants)

    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        #This should likely be equivalent to the maximum workgroup size
        #(or the size needed to activate all cores)
        workgroup_size = 128

    kRestrict0 = lp.split_iname(kRestrict0, "i", workgroup_size,
        outer_tag="g.0", inner_tag="l.0", slabs=(0,1))

    kRestrict0 = lp.add_and_infer_dtypes(kRestrict0, {"indices": np.int32, "uu": fp_format})

    return kRestrict0

def generate_kRestrict1(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kRestrict1 = lp.make_kernel(
        "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
        """
        index := indices[e,i]
        vv[e,d,i] = uu[index + ndof*d]
        """,
        name="kRestrict1",
        target=target,
        assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
        )

    kRestrict1 = lp.fix_parameters(kRestrict1, **constants)

    kRestrict1 = lp.add_and_infer_dtypes(kRestrict1, 
            {"indices": np.int32, "uu": fp_format, "ndof": np.int32})

    return kRestrict1


def generate_kRestrict2(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kRestrict2 = lp.make_kernel(
        "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
        """
        vv[e,d,i] = uu[ncomp*indices[e,i] + d]
        """,
        name="kRestrict2",
        target=target,
        assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
        )

    kRestrict2 = lp.fix_parameters(kRestrict2, **constants)

    kRestrict2 = lp.add_and_infer_dtypes(kRestrict2, {"indices": np.int32, "uu": fp_format})

    return kRestrict2

kRestrict3b = lp.make_kernel(
    "{ [i,j]: 0<=i<ndof, rng1<=j<rngN }",
    """
    <> rng1 = toffsets[i]
    <> rngN = toffsets[i+1]
    for j
        vv[i] = sum(j, uu[indices[j]])
    end
    """,
    name="kRestrict3b",
    target=lp.OpenCLTarget(),
    assumptions="ndof>0 and rng1>=0 and rngN >= rng1")

kRestrict4b = lp.make_kernel(
    "{ [i,d,j]: 0<=i<ndof and 0<=d<ncomp and rng1<=j<rngN }",
    """
    vv[d, i] = sum(j, uu[indices[j]*ncomp + d*elemsize + indices[j]%elemsize])
    """,
    name="kRestrict4b",
    target=lp.OpenCLTarget(),
    assumptions="ndof > 0 and rng1 > 0 and rngN > rng1 and ncomp > 0")

kRestrict5b = lp.make_kernel(
    "{ [i,d,j]: 0<=i<ndof and 0<=d<ncomp and rng1<=j<rngN }",
    """
    vv[i, d] = sum(j, uu[indices[j]*ncomp + d*elemsize + indices[j]%elemsize])
    #vv[i,d] = sum(j, uu[indices[j]*ncomp, d*elemsize, indices[j]%elemsize])
    """,
    name="kRestrict5b",
    target=lp.OpenCLTarget,
    assumptions="ndof > 0 and rng1 > 0 and rngN > rng1 and ncomp > 0"
    )

def generate_kRestrict6(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = ["uu", "vv"]
    if constants=={}:
        kernel_data += ["nelem", "elemsize", "nc"]

    kRestrict6 = lp.make_kernel(
        "{ [e,j,k]: 0<=e<nelem and 0<=i<elemsize and 0<=j<nc }",
        """
        vv[e,j,k] = uu[e,j,k]
        """,
        name="kRestrict6",
        kernel_data=kernel_data,
        assumptions="nelem_x_elemsize_x_ncomp > 0",
        target=target
        )

    kRestrict6 = lp.fix_parameters(kRestrict6, **constants)

    if arch == "AMD_GPU":
        workgroup_size = 64
    elif arch == "NVIDIA_GPU":
        workgroup_size = 32
    else:
        #This should likely be equivalent to the maximum workgroup size
        #(or the size needed to activate all cores)
        workgroup_size = 128

    kRestrict6 = lp.split_iname(kRestrict6, "i", workgroup_size,
        outer_tag="g.0", inner_tag="l.0", slabs=(0,1))

    kRestrict6 = lp.add_and_infer_dtypes(kRestrict6, {"uu": fp_format})
    #kRestrict6 = lp.fix_parameters(kRestrict6, nelem_x_elemsize_x_ncomp=1000)

    return kRestrict6


kRestrict0 = generate_kRestrict0()
code = lp.generate_code_v2(kRestrict0).device_code()
print(kRestrict0)
print()
print(code)
print()

kRestrict1 = generate_kRestrict1()
code = lp.generate_code_v2(kRestrict1).device_code()
print(kRestrict1)
print()
print(code)
print()

kRestrict2 = generate_kRestrict2()
code = lp.generate_code_v2(kRestrict2).device_code()
print(kRestrict2)
print()
print(code)
print() 

'''
kernelList2 = [kRestrict3b]
kernelList3 = [kRestrict4b, kRestrict5b]

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"indices": np.int32, "uu": np.float64, "ndof": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

for k in kernelList3:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"indices": np.int32, "uu": np.float64, "elemsize": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
'''
constants = { "nelem_x_elemsize_x_ncomp":1000 }
kRestrict6 = generate_kRestrict6(constants={})
print(kRestrict6)
print()
print(lp.generate_code_v2(kRestrict6).device_code())
print()
