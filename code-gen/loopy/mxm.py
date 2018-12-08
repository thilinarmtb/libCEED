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

x_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float64)
y_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float64)
z_vec_dev = cl.clrandom.rand(queue, n, dtype=np.float64)
a_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float64)
b_mat_dev = cl.clrandom.rand(queue, (n, n), dtype=np.float64)
x_mat_dev = cl.array.Array(queue, (n, n), dtype=np.float64)
x_mat_host = np.random.rand(n,n)
a_mat_host = np.random.rand(n,n)
x_vec_host = np.random.randn(n).astype(np.float64)
y_vec_host = np.random.randn(n).astype(np.float64)

'''
def test_plain_matrix_mul(ctx_factory):
    ctx = ctx_factory()
    order = "C"

    n = 16#get_suitable_size(ctx)

    for dtype, check, vec_size in [
            (cl_array.vec.float4, check_float4, 4),
            (np.float32, None, 1),
            ]:
        knl = lp.make_kernel(
                "{[i,j,k]: 0<=i,j,k<%d}" % n,
                [
                    "c[i, j] = sum(k, a[i, k]*b[k, j])"
                    ],
                [
                    lp.GlobalArg("a", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("b", dtype, shape=(n, n), order=order),
                    lp.GlobalArg("c", dtype, shape=(n, n), order=order),
                    ],
                name="matmul")

        ref_knl = knl

        knl = lp.split_iname(knl, "i", 16,
                outer_tag="g.0", inner_tag="l.1")
        knl = lp.split_iname(knl, "j", 16,
                outer_tag="g.1", inner_tag="l.0")
        knl = lp.split_iname(knl, "k", 16)
        knl = lp.add_prefetch(knl, "a", ["k_inner", "i_inner"],
                default_tag="l.auto")
        knl = lp.add_prefetch(knl, "b", ["j_inner", "k_inner", ],
                default_tag="l.auto")

        lp.auto_test_vs_ref(ref_knl, ctx, knl,
                op_count=[vec_size*2*n**3/1e9], op_label=["GFlops"], parameters={"n": n}, check_result=check)

test_plain_matrix_mul(cl.create_some_context)

'''
# create
# ------

knl = lp.make_kernel(
        "{ [i]: 0<=i<n }",
        "out[i] = 2*a[i]")
print(knl)

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
mxm = lp.split_iname(mxm, "i", LM, outer_tag="g.0", inner_tag="l.1")
mxm = lp.split_iname(mxm, "j", LN, outer_tag="g.1", inner_tag="l.0")
mxm = lp.split_iname(mxm, "k", LO, inner_tag="unr")
mxm = lp.add_prefetch(mxm, "A", ["k_inner", "i_inner"], default_tag="l.auto")
mxm = lp.add_prefetch(mxm, "X", ["j_inner", "k_inner", ], default_tag="l.auto")




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
evt, (B,) = mxm(queue, A=a_mat_host, X=x_mat_host)#, B=b_mat_dev)
#evt, (out,) = knl(queue, a=a)

#print(B)
#print(a_mat_host @ x_mat_host)
print((B == a_mat_host @ x_mat_host).all())
