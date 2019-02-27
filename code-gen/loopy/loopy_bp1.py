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

    <> dummy = ctx[0] # Need to figure out how to remove

    <> val1 = in[i + iOf7[2]]*reduce(sum,d, s*in[i+os0] * (in[i+os1]*in[i+os2] - in[i+os3]*in[i+os4]))
    out[i + oOf7[0]] = val1
    out[i + oOf7[1]] = val1 * w 
    """,
    name="masssetupf",
    assumptions="Q > 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
for os in ["e","f","os0","os1","os2","os3","os4","s"]:
    masssetupf = lp.precompute(masssetupf, os, "d")
masssetupf = lp.precompute(masssetupf, "w")
masssetupf = lp.prioritize_loops(masssetupf,"i,d")
masssetupf = lp.tag_inames(masssetupf, {"d": "unr"})
#masssetupf = lp.split_iname(masssetupf, "i", 4, outer_tag="g.0")

#print(setup)    

massf = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    o_os := i + oOf7[0]
    i_os0 := i + iOf7[0]
    i_os1 := i + iOf7[1]
    <> dummy = ctx[0]
    out[o_os] = in[i_os0] + in[i_os1]
    #out[i+oOf7[0]] = in[i+iOf7[0]] * in[i+iOf7[1]]
    """,
    name="massf",
    assumptions="Q > 0",
    kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
    target=lp.OpenCLTarget()
    )
print(massf)
massf = lp.split_iname(massf, "i", 8, outer_tag="g.0", inner_tag="ilp", slabs=(0,1))
#massf = lp.split_array_axis(massf, "out,in", axis_nr=0, count=4)
#massf = lp.tag_array_axes(massf, "out, in", "C,vec")

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
