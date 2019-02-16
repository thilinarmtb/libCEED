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

setup = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    out[i + oOf7[0]] = in[i + iOf7[0]]
    """,
    name="setup",
    assumptions="Q >= 0",
    target=lp.OpenCLTarget()
    )
#print(setup)    

mass = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    out[i + oOf7[0]] = scale[4] * in[i + iOf7[0]] * in[i + iOf7[1]]
    """,
    name="mass",
    assumptions="Q >= 0",
    target=lp.OpenCLTarget()
    )
#print(mass)

kernelList1 = [setup]
kernelList2 = [mass]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "oOf7": np.int32,
        "iOf7": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "oOf7": np.int32,
        "iOf7": np.int32,
        "scale": np.float64
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
