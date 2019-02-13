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

ellipticAxHex3D_Ref3D1= lp.make_kernel(
    "{ [e,n,i,j,k]: 0<=e<Nelements and 0<=n,i,j,k<p_Ngll}",
    """
    for e,i,j,k
        <> base = e*p_Ngeo*p_Np + k*p_Ngll*p_Ngll + j*p_Ngll + i
        <> GwJ = ggeo[base + p_GWJID*p_Np]
        <> qr = simul_reduce(sum, n, D[i,n]*q[n + j*p_Ngll + k*p_Ngll*p_Ngll + e*p_Np])
        <> qs = simul_reduce(sum, n, D[j,n]*q[i + n*p_Ngll + k*p_Ngll*p_Ngll + e*p_Np])
        <> qt = simul_reduce(sum, n, D[k,n]*q[i + j*p_Ngll + n*p_Ngll*p_Ngll + e*p_Np])
       
        <> base1 = i + j*p_Ngll + k*p_Ngll*p_Ngll + e*p_Np
        Aq[base1] = GwJ*lambda*q[base1]
    end
    """,
    name="ellipticAxHex3D_Ref3D1",
    assumptions="Nelements > 0 and p_Ngll > 0 ",
    target=lp.OpenCLTarget()
    )
print(ellipticAxHex3D_Ref3D1)    

kernelList1 = [ellipticAxHex3D_Ref3D1]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "D": np.float64,
        "q": np.float64,
        "lambda": np.float64,
        "ggeo": np.float64,
        "p_Np": np.int32,
        "p_Ngeo": np.int32,
        "p_GWJID": np.int32 
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()


