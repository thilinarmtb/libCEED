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

kRestrict = lp.make_kernel(
    [ "{ [e,j,k,d]: 0<=e<nblk and 0<=j<blksize and 0<=k<elemsize and 0<=d<ncomp}",
      "{ [jj]: 0<=jj<jmax }"
        ],
    """
    #<> index1
    #<> index2
    for e, k, d
        if tmode == CEED_NOTTRANSPOSE
            for j 
                with {id_prefix=id1}
                if not use_ind
                    # No indices provided, Identity Restriction
                    <> val1 = e*blksize + j
                    <> val2 = nelem - 1
                    <> index1 = e*elemsize*ncomp*blksize + k*ncomp*blksize + d*blksize + j 
                    if val1 < val2
                        <> index2 = e*blksize*ncomp*elemsize + j*ncomp*elemsize + k*ncomp + d
                    else
                        index = val2*ncomp*elemsize + k*ncomp + d
                    end
                else
                    index1 = e*blksize*elemsize*ncomp + d*blksize*elemsize + k*blksize + j
                    if lmode == CEED_NOTTRANSPOSE
                        index2 = indices[e*blksize*elemsize + j*elemsize + k] + ndof*d
                    else
                        index2 = indices[e*blksize*elemsize + j*elemsize + k]*ncomp + d
                    end
                end
                end
                vv[index] = uu[index2] {dep=id1}
            end
        else
            val1 = nelem - e
            <> jmax = if(blksize < val1, blksize, val1)
            for jj
                with {id_prefix=id2}
                if not use_ind
                    # Restriction from evector to lvector
                    # Performing v += r^T * u
                    # No indices provided, Identity Restriction

                    index1 = e*elemsize*ncomp*blksize + jj*ncomp*elemsize + k*ncomp + d
                    index2 = e*elemsize*ncomp*blksize + k*ncomp*blksize + blksize*d + jj
                else
                    # Indices provided, standard or blocked restriction
                    # uu has shape [elemsize, ncomp, nelem]
                    # vv has shape [ndof, ncomp] 
                    # Iteration bound set to discard padding elements

                    index2 = e*ncomp*blksize*elemsize + d*blksize*elemsize + k*blksize + jj
                    if not lmode == CEED_NOTTRANSPOSE
                        index1 = indices[e*elemsize*blksize + k*blksize + jj] + ndof*d 
                    else
                        index1 = indices[e*elemsize*blksize + k*blksize + jj]*ncomp + d
                    end 
                end
                end
                vv[index1] = vv[index1] + uu[index2] {dep=id2*}
            end
        end
    end
    """,
    name = "kRestrict",
    assumptions="blksize > 0 and elemsize > 0 and ncomp > 0 and nblk > 0",
    target=lp.OpenCLTarget()
)
print(kRestrict)

'''
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
    "{ [i,j]: 0<=i<ndof, rng1<=j<rngN }",
    """
    <> rng1 = toffsets[i]
    <> rngN = toffsets[i+1]
    for j
        vv[i] = sum(j, uu[indices[j]])
    end
    """,
    name="kRestrict3b",
    target=lp.OpenCLTarget(),
    assumptions="ndof>0 and rng1>=0 and rngN >= rng1")

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


kernelList2 = [kRestrict1, kRestrict3b]
kernelList3 = [kRestrict4b, kRestrict5b]
'''

kernelList1 = [kRestrict]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "uu": np.float64,
        "vv": np.float64,
        "indices": np.int32,
        "nelem": np.int32,
        "tmode": np.int32,
        "use_ind": np.int32,
        "lmode": np.int32,
        "ndof": np.int32,
        "CEED_NOTTRANSPOSE": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)
    #print(k)


