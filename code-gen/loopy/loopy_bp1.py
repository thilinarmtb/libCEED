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

# Currently works only for 3D
def generate_masssetupf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    masssetupf = lp.make_kernel(
        "{ [i,d,dd]: 0<=i<Q and 0<=d,dd<3 }",
        """
        D := 3
        v(a, b) := in[(a*D + b)*Q + i + iOf7[1]]

        if false
            ctx[0] = 0
        end

        det := in[iOf7[2] + i] * (
                   v(0,0) * (v(1,1)*v(2,2) - v(1,2)*v(2,1))
                 - v(0,1) * (v(1,0)*v(2,2) - v(1,2)*v(2,0))
                 + v(0,2) * (v(1,0)*v(2,1) - v(1,1)*v(2,0)))

        sum := v(0,0)**2 + v(0,1)**2 + v(0,2)**2

        out[oOf7[0] + i] = det
        out[oOf7[1] + i] = det * sqrt(sum) 
        """,
        name="masssetupf",
        assumptions="Q > 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=target
        )

    masssetupf = lp.add_and_infer_dtypes(masssetupf, {
        "ctx": np.int32, "in": fp_format,
        "oOf7": np.int32,"iOf7": np.int32  
        })
    
    masssetupf = lp.tag_inames(masssetupf, {"d": "unr", "dd": "unr"})

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

        out[o_os] = in[i_os0] + in[i_os1]
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
        "ctx": np.int32,
        "in": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32 
    })

    return massf

masssetupf = generate_masssetupf()
code = lp.generate_code_v2(masssetupf).device_code()
print(masssetupf)
print()
print(code)

massf = generate_massf()
code = lp.generate_code_v2(massf).device_code()
print(massf)
print()
print(code)

'''
kernelList1 = [masssetupf, massf]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
'''
