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
def generate_masssetupf(arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    masssetupf = lp.make_kernel(
        "{ [i,d,dd]: 0<=i<Q and 0<=d,dd<3 }",
        """
        val := in[dd*Q + iOf7[0]]**2
        w := sqrt(reduce(sum, dd, val))
        e := (d + 1) % 3
        f := (d + 2) % 3
        os0 := iOf7[1] + (0 + d)*Q 
        os1 := iOf7[1] + (3 + e)*Q
        os2 := iOf7[1] + (6 + f)*Q
        os3 := iOf7[1] + (6 + e)*Q
        os4 := iOf7[1] + (3 + f)*Q
        s := -1**d

        if false
            <> dummy = ctx[0] # Need to figure out how to remove
        end

        <> val1 = in[i + iOf7[2]]*reduce(sum,d, s*in[i+os0] * (in[i+os1]*in[i+os2] - in[i+os3]*in[i+os4]))
        out[i + oOf7[0]] = val1
        out[i + oOf7[1]] = val1 * w 
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
        masssetupf = lp.split_iname(masssetupf, "i", 64, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    elif arch == "NVIDIA_GPU":
        masssetupf = lp.split_iname(masssetupf, "i", 32, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    else:
        masssetupf = lp.split_iname(masssetupf, "i", 128, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
 
    return masssetupf


def generate_massf(arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    massf = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        o_os := i + oOf7[0]
        i_os0 := i + iOf7[0]
        i_os1 := i + iOf7[1]
        if false
            <> dummy = ctx[0] # Compiler will hopefully remove
        end

        out[o_os] = in[i_os0] + in[i_os1]
        """,
        name="massf",
        assumptions="Q > 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=target
        )

    if arch == "AMD_GPU":
        massf = lp.split_iname(massf, "i", 64, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    elif arch == "NVIDIA_GPU":
        massf = lp.split_iname(massf, "i", 32, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    else:
        massf = lp.split_iname(massf, "i", 128, outer_tag="g.0", inner_tag="l.0", slabs=(0,1))   
    
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
