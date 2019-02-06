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

#ctx = cl.create_some_context(interactive=True)
#queue = cl.CommandQueue(ctx)


#x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
#y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
#z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
#a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
#b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
#x_mat_dev = cl.array.Array(queue, (n, n), dtype=np.float32)
#x_mat_host = np.float32(np.random.rand(n,n))
#a_mat_host = np.float32(np.random.rand(n,n))
#x_vec_host = np.random.randn(n).astype(np.float32)
#y_vec_host = np.random.randn(n).astype(np.float32)
# create
# ------

setupf = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    out[oOf7_0 + i] = in[iOf7_0 + i]
    """,
    name="setupf",
    assumptions="Q >= 0",
    target=lp.OpenCLTarget()
    )
print(setupf)    

# Something is messed up here
massf = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    for i
        <> index0 = iOf7_0 + i
        <> index1 = iOf7_1 + i
        out[oOf7_0 + i] = in[index0]*in[index1]
    end
    """,
    name="massf",
    assumptions="Q >= 0",
    target=lp.OpenCLTarget()
    )
print(massf)

kernelList1 = [setupf]
kernelList2 = [massf]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "out": np.float64,
        "oOf7_0": np.int32,
        "iOf7_0": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "out": np.float64,
        "oOf7_0": np.int32,
        "iOf7_0": np.int32,
        "iOf7_1": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

