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

setup = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    <> dummy = ctx[0] # Need to figure out how to remove
    out[i + oOf7[0]] = in[i + iOf7[0]]
    """,
    name="setup",
    assumptions="Q >= 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
#print(setup)    

mass = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    <> dummy = ctx[0]
    out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]]
    """,
    name="mass",
    assumptions="Q >= 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
#print(mass)

kernelList1 = [setup, mass]

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
