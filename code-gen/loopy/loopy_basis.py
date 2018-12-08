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

kZero = lp.make_kernel(
    "{ [e,i]: 0<=e<nelem and 0<=i<vsize }",
    """
    v[e*nc*elemsize + i] = 0
    """,
    assumptions="nelem > 0 and vsize > 0"
    )
print(kZero)

kCeedTensorContract = lp.make_kernel(
    "{ [a,j,b,c]: 0<=a<A and 0<=j<J and 0<=b<B and 0<=c<C }",
    """
    v[a,j,c] = Zero*v[a,j,c] + t[j*stride0 + b*stride1] * u[a,b,c]
    """,
    assumptions="A>0 and B>0 and C>0 and J>0")
print(kCeedTensorContract)

#This is wrong, d_v part needs to be inside all loops
kWeight = lp.make_kernel(
    ["{ [e,d,j]: 0<=e<nelem and 0<=d<dim and 0<=j<Q }",
    "{ [i]: 0<=i<pre }",
    "{ [k]: 0<=k<post }"],
    """
    for d, e
        <> v_offset = e*QnD*nc*(dim + 2) + (QnD*nc + QnD*nc*dim)
        <> pre = pow(Q, dim-d-1)
        <> post = pow(Q, d)
        <> xs =((i*Q + j)*post + k) + v_offset {id=calc} 
        d_v[xs] = qweight1d[j]*if(d==0, 1, d_v[xs]) 
    end
    """, 
    assumptions="nelem>0 and dim>0 and Q>0")
#kWeight = lp.prioritize_loops(kWeight, "i,k,j")
print(kWeight)

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
