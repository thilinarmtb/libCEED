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
    ["{ [a,j,b]: 0<=a<A and 0<=j<J and 0<=b<B }",
     "{ [c]: 0<=c<C}",
     "{ [c_wxs]: wxs_os<=c_wxs<C+wxs_os}",
     "{ [c_rxs]: rxs_os<=c_rxs<C+rxs_os}"
    ],
    """
    <> tstride0 = if(transpose, 1, B)
    <> tstride1 = if(transpose, J, 1)
    #wxs = ((a*J+j)*C+c) + wxs_os
    #rxs = ((a*B+b)*C+c) + rxs_os
    #v[wxs] = Add*v[wxs] + t[j*stride0 + b*stride1] * u[rxs]
    #v[a,j,c_wxs] = Add*v[a,j,c_wxs] + t[j*stride0 + b*stride1] * u[a,b,c_rxs]
    
    # Alternative above, would be best if loopy could generate with loop inside
    if Add 
        v[a,j,c_wxs] = v[a,j,c_wxs] + t[j*stride0 + b*stride1] * u[a,b,c_rxs]
    else
        v[a,j,c_wxs] = t[j*stride0 + b*stride1] * u[a,b,c_rxs]
    end
    """,
    assumptions="A>0 and B>0 and C>0 and J>0")
kCeedTensorContract = lp.set_instruction_priority(kCeedTensorContract, "id:conditional", 10)
print(kCeedTensorContract)

kInterp = lp.make_kernel(
    ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
    "{ [a,j,c]: 0<=a<pre and 0<=j<P and 0<=c<Q }"],
    """
    <> P = if(transpose, Q1d, P1d)
    <> Q = if(transpose, P1d, Q1d)
 
    if(transpose)
        <>P=Q1d
                 
    else
    
    end

    

    """,
    assumptions="nelem>0 and dim>0")

'''
kInterp = lp.make_kernel(
    ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
     "{ [a,j,c]: 0<=a<pre and 0<=j<P and 0<=c<Q }"],
    """
    <> P = if(transpose, Q1d, P1d)
    <> Q = if(transpose, P1d, Q1d)
    for e
        # Calculate offsets to data and temporary arrays
        <> t_offset = e*tmpSz
        <> u_offset = e*nc*elemsize
        <> v_offset = e*QnD*nc*(dim+2)
        <> d_u_offset = if(transpose, v_offset, u_offset)
        <> d_v_offset = if(transpose, u_offset, v_offset)
        <> wxs_os = if(d==0,     d_u_offset, t_offset)
        <> rxs_os = if(d==dim-1, d_v_offset, t_offset) 

        for d
            <> Add = (transpose & (d == dim - 1))
            <> pre = ndof*pow(P, dim-1-d)
            <> post = pow(Q, d) 

            #This is problematic              
            <> u = if(d==0,     d_u[n], tmp0[n])
            <> v = if(d==dim-1, d_v, tmp1)

            d_v[(a*Q + j)*post + c + wxs_os] = f[(a*Q + j)*post + c + wxs_os]
            #d_v[(a*J + j)*C + c + wxs_os] = f[(a*J + j)*C + c + wxs_os]
            #d_v[a,j,c+wxs_os] = f[a,j,c+wxs_os]   
        end

        d_v_offset = d_v_offset + if(not transpose, nqpt, 0)
        d_u_offset = d_u_offset + if( transpose, nqpt, 0)
    end
    """,
    assumptions="nelem>0 and dim>0")
print(kInterp)
'''
#data_flow = [ (v,0,1), (f,1,0), (u,0,1)
#    (a,0,1),
#    (j,0,1),
#    (c,0,1),
#    (wxs_os,0,1),
#    (rxs_os,0,1),
#    (v,0,1),
#    (u,0,1)
#]
#kIntern_fused = lp.fuse_kernels((kInterp, kCeedTensorContract), data_flow=data_flow)
#print(kInterk_fused)

#This is wrong, d_v part needs to be inside all loops
kWeight = lp.make_kernel(
    ["{ [e,d]: 0<=e<nelem and 0<=d<dim}",
#    "{ [i]: 0<=i<pre }",
#    "{ [k]: 0<=k<post }"],
    "{ [dInd]: dInd=d+2}",
    "{ [kInd]: kInd=k+v_shift}",
    "{ [i,j,k]: 0<=i,j,k<Q }"],
    """
    <> v_shift = (QnD*nc + QnD*nc*dim)
#   <> v_offset = e*QnD*nc*(dim + 2) + v_shift
#   <> post = pow(Q, d)
#   <> pre = pow(Q, dim-d-1) 
#   <> xs =((i*Q + j)*post + k) + v_offset
#   <> xs =((i*Q + j)*Q + k) + v_offset

    for k
    #Scalars only right now, need to incorpate nc otherwise
    #<> jVal = qweight1d[j]
    if d == 0
        d_v[e,dInd,i,j,kInd] = qweight1d[j]
    else
        d_v[e,dInd,i,j,kInd] = qweight1d[j]*d_v[e,dInd,i,j,kInd]
    end
    end
    """, 
    assumptions="nelem>0 and dim>0 and Q>0")
kWeight = lp.prioritize_loops(kWeight,"e,d,i,j,k")
print(kWeight)

mxm = lp.make_kernel(
      "{ [i,j,k,ii,kk]: 0<=i<m and 0<=j<n and 0<=k<o}",
      """
      B[i, j] = sum(k, A[i, k]*X[k, j])
      """,assumptions="n mod 16 = 0 and m mod 16 = 0 and o mod 16 = 0 and n,m,o >= 16")
#mxm = lp.set_options(mxm, "write_cl")
print(mxm)



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
