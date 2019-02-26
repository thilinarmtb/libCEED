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

#ctx = lp.ArrayArg("ctx", dtype=np.int32, address_space=lp.auto)

masssetupf = lp.make_kernel(
    "{ [i, d,dd]: 0<=i<Q and 0<=d,dd<3 }",
    """
    <> dummy = ctx[0] # Need to figure out how to remove
    for i
    <> offset = iOf7[1] + i
    for dd
        <> e = (dd + 1) % 3 
        <> f = (dd + 2) % 3
        <> index0 = offset + (0 + dd)*Q 
        <> index1 = offset + (3 + e)*Q
        <> index2 = offset + (6 + f)*Q
        <> index3 = offset + (6 + e)*Q
        <> index4 = offset + (3 + f)*Q
        
        <> detv = (-1**dd)*in[index0] * \
            (in[index1]*in[index2] - in[index3]*in[index4])
    end

    <> det = sum(dd, detv)
    <> val1 = det*in[i + iOf7[2]]
    out[i + oOf7[0]] = val1
    out[i + oOf7[1]] = val1 * sqrt(sum(d, in[d*Q + i + iOf7[0] ]))

    end
    """,
    name="masssetupf",
    assumptions="Q >= 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
#print(setup)    

massf = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    <> dummy = ctx[0]
    out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]]
    """,
    name="massf",
    assumptions="Q >= 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
#print(mass)

kernelList1 = [masssetupf, massf]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "ctx": np.int32,
        "in": np.float64,
        "oOf7": np.int32,
        "iOf7": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
