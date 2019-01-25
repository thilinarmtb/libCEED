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

#ctx = cl.create_some_context(interactive=False)
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

kRestrict0 = lp.make_kernel(
    "{ [i]: 0<=i<nelem_x_elemsize}",
    """
    vv[i] = uu[indices[i]]
    """,
    name="kRestrict0",
    assumptions="nelem_x_elemsize > 0",
    target=lp.OpenCLTarget() #Don't want to hardcode this, but will do for now 
    )

kRestrict1 = lp.make_kernel(
    "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
    """
    vv[e,d,i] = uu[indices[e,i] + ndof*d]
    """,
    name="kRestrict1",
    target=lp.OpenCLTarget(),
    assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
    )

kRestrict2 = lp.make_kernel(
    "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
    """
    vv[e,d,i] = uu[ncomp*indices[e,i] + d]
    """,
    name="kRestrict2",
    target=lp.OpenCLTarget(),
    assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
    )

kRestrict3b = lp.make_kernel(
    "{ [i,j]: 0<=i<ndof and rng1<=j<rngN }",
    """
    vv[i] = sum(j, uu[indices[j]])
    """,
    name="kRestrict3b",
    target=lp.OpenCLTarget(),
    assumptions="ndof>0 and rng1>0 and rngN > rng1")

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
    target=lp.OpenCLTarget(),
    assumptions="ndof > 0 and rng1 > 0 and rngN > rng1 and ncomp > 0"
    )

kernelList1 = [kRestrict0, kRestrict2]
kernelList2 = [kRestrict1, kRestrict3b]
kernelList3 = [kRestrict4b, kRestrict5b]


for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"indices": np.int32, "uu": np.float64})
    code = lp.generate_code_v2(k).device_code()
    print(code)
    #print(k)

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"indices": np.int32, "uu": np.float64, "ndof": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)

for k in kernelList3:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"indices": np.int32, "uu": np.float64, "elemsize": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)
