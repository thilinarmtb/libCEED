import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

LN = 8
LM = 8
LO = 8

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

#n = 15 * 10**6
#a = cl.array.arange(queue, n, dtype=np.float32)
n = 16*16

x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float32)
a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float32)
x_mat_dev = cl.array.Array(queue, (n, n), dtype=np.float32)
x_mat_host = np.float32(np.random.rand(n,n))
a_mat_host = np.float32(np.random.rand(n,n))
x_vec_host = np.random.randn(n).astype(np.float32)
y_vec_host = np.random.randn(n).astype(np.float32)
# create
# ------

kRestrict0 = lp.make_kernel(
    "{ [i]: 0<=i<nelem_x_elemsize}",
    """
    vv[i] = uu[indices[i]]
    """,
    assumptions="nelem_x_elemsize > 0"
    )
print(kRestrict0)

kRestrict1 = lp.make_kernel(
    "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
    """
    vv[i + elemsize*(d + ncomp*e)] = uu[indices[i + elemsize*e] + ndof*d]

#    Use more readable indexing in the future
#    vv[i+elemsize*(d+ncomp*e)] = uu[indices[e][i]+ndof*d]
#    vv[][][i]
    """,
    assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
    )
print(kRestrict1)

kRestrict2 = lp.make_kernel(
    "{ [e,d,i]: 0<=e<nelem and 0<=d<ncomp and 0<=i<elemsize }",
    """
    vv[i+elemsize*(d+ncomp*e)] = uu[ncomp*indices[i+elemsize*e] + d]
    """,
    assumptions="nelem > 0 and ncomp > 0 and elemsize > 0"
    )
print(kRestrict2)

kRestrict3b = lp.make_kernel(
    "{ [i,j]: 0<=i<ndof and rng1<=j<rngN }",
    """
    vv[i] = sum(j, uu[tindices[j]])
    """,
    assumptions="ndof>0 and rng1>0 and rngN > rng1"
    )
print(kRestrict3b)

kRestrict4b = lp.make_kernel(
    "{ [i,d,j]: 0<=i<ndof and 0<=d<ncomp and rng1<=j<rngN }",
    """
    vv[d, i] = sum(j, uu[((tindices[j]/elemsize)*ncomp + d)*elemsize + (tindices[j]%elemsize)])
    """,
    assumptions="ndof > 0 and rng1 > 0 and rngN > rng1 and ncomp > 0"
    )
print(kRestrict4b)

kRestrict5b = lp.make_kernel(
    "{ [i,d,j]: 0<=i<ndof and 0<=d<ncomp and rng1<=j<rngN }",
    """
    vv[i, d] = sum(j, uu[((tindices[j]/elemsize)*ncomp + d)*elemsize + (tindices[j]%elemsize)])
    """,
    assumptions="ndof > 0 and rng1 > 0 and rngN > rng1 and ncomp > 0"
    )
print(kRestrict5b)



mxm = lp.make_kernel(
      "{ [i,j,k,ii,kk]: 0<=i<m and 0<=j<n and 0<=k<o}",
      """
      B[i, j] = sum(k, A[i, k]*X[k, j])
      """,assumptions="n mod 16 = 0 and m mod 16 = 0 and o mod 16 = 0 and n,m,o >= 16")
#c[i, j] = sum(k, a[i, k]*b[k, j])
#mxm = lp.prioritize_loops(mxm, "k,kk,i,ii,j") #Based on whether or not ii and kk is used
#mxm = lp.prioritize_loops(mxm, "k,kk,j,i,ii")
#mxm = lp.prioritize_loops(mxm, "i, ii, k, kk, j");
mxm = lp.set_options(mxm, "write_cl")

#Github optimizations
'''
mxm = lp.split_iname(mxm, "i", LM, outer_tag="g.0", inner_tag="l.1")
mxm = lp.split_iname(mxm, "j", LN, outer_tag="g.1", inner_tag="l.0")
mxm = lp.split_iname(mxm, "k", LO, inner_tag="unr")
mxm = lp.add_prefetch(mxm, "A", ["k_inner", "i_inner"], default_tag="l.auto")
mxm = lp.add_prefetch(mxm, "X", ["j_inner", "k_inner", ], default_tag="l.auto")
'''

print(mxm)

#Split into cache-line sized chunks
#mxm = lp.split_iname(mxm, "i", 8)
#mxm = lp.split_iname(mxm, "k", 8)
#mxm = lp.split_iname(mxm, "j", 8)

#mxm = lp.add_prefetch("mxm", "B", ["i_inner", "k_inner"])

#mxm = lp.split_iname(mxm, "i_inner", 2)
#mxm = lp.split_iname(mxm, "k_inner", 2)
#mxm = lp.split_iname(mxm, "j_inner", 2)

#mxm = lp.tag_inames(mxm, dict(i_outer="g.0", i_inner="l.0"))

#mxm = lp.split_iname(mxm, "j", 8, inner_tag="unr")
#mxm = lp.split_iname(mxm, "ii", 8)
#mxm = lp.split_iname(mxm, "kk", 8)
#mxm = lp.tag_inames(mxm, dict(j_inner="unr"))

#From Fortran mxm example
#mxm = lp.split_iname(mxm, "i", 8, outer_tag="g.0", inner_tag="l.1")
#mxm = lp.split_iname(mxm, "j", 8, outer_tag="g.1", inner_tag="l.0")
#mxm = lp.split_iname(mxm, "k", 8)
#mxm = lp.extract_subst(mxm, "a_acc", "a[i1,i2]", parameters="i1, i2")
#mxm = lp.extract_subst(mxm, "b_acc", "b[i1,i2]", parameters="i1, i2")
#mxm = lp.precompute(mxm, "a_acc", "k_inner,i_inner", default_tag="l.auto")
#mxm = lp.precompute(mxm, "b_acc", "j_inner,k_inner", default_tag="l.auto")
#print(mxm)



#code, _ = lp.generate_code(mxm)
#print(code)

# transform
# ---------
#knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

# execute
# -------
#evt, (B,) = mxm(queue, A=a_mat_host, X=x_mat_host)#, B=b_mat_dev)
#evt, (out,) = knl(queue, a=a)

#print(B)
#print(a_mat_host @ x_mat_host)
#print((B - a_mat_host @ x_mat_host).max())
