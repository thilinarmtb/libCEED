import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

LN = 8
LM = 8
LO = 8
NE = 2

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

ctx = cl.create_some_context(interactive=True)
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
x_mat_host = np.float32(np.random.rand(NE, n,n))
a_mat_host = np.float32(np.random.rand(n,n))
x_vec_host = np.random.randn(n).astype(np.float32)
y_vec_host = np.random.randn(n).astype(np.float32)


# create
# ------
mxm = lp.make_kernel(
      "{ [i,j,k,e]: 0<=i<m and 0<=j<n and 0<=k<o and 0<=e<NE}",
      """
      for e 
        B[e, i, j] = sum(k, A[i, k]*X[e, k, j])
      end
   
      """,assumptions="n mod 16 = 0 and m mod 16 = 0 and o mod 16 = 0 and n,m,o >= 16 and NE > 0")

#X[NE, ,] = sum(k, A[i,k]*B[NE,, ]) 
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


#code, _ = lp.generate_code(mxm)
#print(code)

# transform
# ---------
#knl = lp.split_iname(knl, "i", 128, outer_tag="g.0", inner_tag="l.0")

# execute
# -------
evt, (B,) = mxm(queue, A=a_mat_host, X=x_mat_host)#, B=b_mat_dev)
#evt, (out,) = knl(queue, a=a)

#print(B)
#print(a_mat_host @ x_mat_host)
#print((B - a_mat_host @ x_mat_host).max())
