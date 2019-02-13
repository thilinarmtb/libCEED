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
    ["{ [i,j,k]: 0<=i<Q and 0<=j,k<3}",
     "{ [kk,jj]: 0<=jj,kk<3}",
     "{ [kkk,jjj,lll]: 0<=kkk,lll,jjj<3}" ],
    """
    # For reference
    #    *J = in + iOf7[0],
    #    *qw = in + iOf7[1];
    #    *qd = out + oOf7[0];

    for i
        <> iind0 = i + iOf7_0
        <> oind0 = i + oOf7_0
        for jj,kk
            <> J[jj,kk] = in[3*Q*jj + Q*kk + iind0]
        end
        for j,k
            <> m = (j + 1) % 3
            <> n = (j + 2) % 3
            <> o = (k + 1) % 3
            <> p = (k + 2) % 3
            <> A[k,j] = J[n,p]*J[m,o] - J[n,o]*J[m,p]
        end
        <> w = in[iOf7_1 + i] / sum(k, J[k,0]*A[0,k])
        for jjj,kkk
            out[3*Q*jjj + Q*kkk + oind0] = sum(lll, A[jjj,lll]*A[kkk,lll])
        end

        #for j,k,l
        #    if k >= j
        #        qd[j,k] = out[j,k,i+oOf7_0] + A[j,l]*A[k,l]
        #    end
        #end
    end 
    """,
    name="setupf",
    assumptions="Q >= 1",
    target=lp.OpenCLTarget()
    )
print(setupf)    

diffusionf = lp.make_kernel(
    "{ [i,j]: 0<=i<Q and 0<=j<3 }",
    """
    #    For Reference
    #    *ug = in  + iOf7[0],
    #    *qd = in  + iOf7[1];
    #    *vg = out + oOf7[0];

    for i, j
        <> index = Q*j + i
        out[index + oOf7_0] = in[index + iOf7_1] * in[index + iOf7_0]
    end
    """,
    name="diffusionf",
    assumptions="Q > 0",
    target=lp.OpenCLTarget()
    )
print(diffusionf)

kernelList1 = [setupf]
kernelList2 = [diffusionf]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "oOf7_0": np.int32,
        "iOf7_0": np.int32,
        "iOf7_1": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "in": np.float64,
        "oOf7_0": np.int32,
        "iOf7_0": np.int32,
        "iOf7_1": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()

