import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

'''
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
'''

# ------

kZero = lp.make_kernel(
    "{ [e,i]: 0<=e<nelem and 0<=i<vsize }",
    """
    v[e*(nc*elemsize) + i] = 0
    """,
    name="kZero",
    assumptions="nelem > 0 and vsize > 0",
    target=lp.OpenCLTarget()
)
#z_tst_dat = np.float32(np.random.rand(2,64))
#kZero = lp.set_options(kZero, "write_cl")
#evt, (out,) = kZero(queue, v=z_tst_dat,nc=1,nelem=6,vsize=64, elemsize=64)

kCeedTensorContract = lp.make_kernel(
    ["{ [a,b,c,j]: 0<=a<A and 0<=b<B and 0<=c<C and 0<=j<J }"
     #"{ [c_wxs]: wxs_os<=c_wxs<C+wxs_os}",
     #"{ [c_rxs]: rxs_os<=c_rxs<C+rxs_os}"
    ],
    """
    <> tstride0 = if(transpose, 1, B)
    <> tstride1 = if(transpose, J, 1)
    for a,j,c
        <> wxs = ((a*J+j)*C+c) + wxs_os
        <> rxs = ((a*B+b)*C+c) + rxs_os
        #v[wxs] = if(Add, v[wxs], 0) + t[j*stride0 + b*stride1] * u[rxs]
        #All choose the same so this flow control should be fine
        if Add 
            v[wxs] = v[wxs] + t[j*stride0 + b*stride1] * u[rxs]
        else
            v[wxs] = t[j*stride0 + b*stride1] * u[rxs]
        end 
    end
    #Better for GPU?
    #v[a,j,c_wxs] = if(Add,v[a,j,c_wxs],0) + t[j*stride0 + b*stride1] * u[a,b,c_rxs]
  
    #Better for CPU?
    #if Add 
    #    v[a,j,c_wxs] = v[a,j,c_wxs] + t[j*stride0 + b*stride1] * u[a,b,c_rxs]
    #else
    #    v[a,j,c_wxs] = t[j*stride0 + b*stride1] * u[a,b,c_rxs]
    #end

    """,
    name = "kCeedTensorContract",
    target=lp.OpenCLTarget(),
    assumptions="A>0 and B>0 and C>0 and J>0"
    )
print(kCeedTensorContract)


kInterp = lp.make_kernel(
    ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
     "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
    """
    <> P = if(transpose, Q1d, P1d)
    <> Q = if(transpose, P1d, Q1d)
    for e
        # Calculate offsets to data and temporary arrays
        #<> t_offset = e*tmpSz
        <> u_offset = e*nc*elemsize
        <> v_offset = e*QnD*nc*(dim+2)
        <> d_u_offset = if(transpose, v_offset, u_offset)
        <> d_v_offset = if(transpose, u_offset, v_offset)
        #<> wxs_os = if(d==0,     d_u_offset, t_offset)
        #<> rxs_os = if(d==dim-1, d_v_offset, t_offset) 
 
        #<> pre = ndof
        #<> post = 1 
        #for d
        #    pre = pre*P   
        #end

        with {id_prefix=d_loop}
        for d
            #<> Add = (transpose & (d == dim - 1))
            #Could also use prefilled array, may be more efficient
            <> pre = ndof*(P**(dim-1-d))
            <> post = Q**d 

            for a,b,c,j
                <> indw = ((a*Q+j)*post + c) 
                <> indr = ((a*P+b)*post + c) 
                <> rxs = indr + d_v_offset
                <> wxs = indw + d_u_offset

                <> tstride0 = if(transpose, 1, P)
                <> tstride1 = if(transpose, Q, 1)

                if d == 0
                    if d == dim - 1
                        d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * d_u[rxs] 
                    else
                        tmp1[indw] = interp1d[j*stride0 + b*stride1] * d_u[rxs]                 
                    end
                elif d == dim - 1
                    if d%2 == 0
                        d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp1[indr]
                    else    
                        d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp0[indr]
                    end
                elif d%2 == 0
                    tmp0[indw] = interp1d[j*stride0 + b*stride1] * tmp1[indr]
                else
                    tmp1[indw] = interp1d[j*stride0 + b*stride1] * tmp0[indr]
                end
            end

         #   pre = pre / P
         #   post = post * Q

        end
        end

        with {dep=d_loop*}
        if transpose
            d_u_offset = d_u_offset + nqpt
        else
            d_v_offset = d_v_offset + nqpt
        end
        end
    end
    """,
    name="kInterp",
    target=lp.OpenCLTarget(),
    assumptions="nelem>0 and dim>0 and pre>0 and post>0 and P>0 and Q>0")
print(kInterp)

#Surely there is a way to combine kGrad and kInterp. They are mostly identical codewise
kGrad = lp.make_kernel(
    ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<dim }",
     "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
    """
    <> P = if(transpose, Q1d, P1d)
    <> Q = if(transpose, P1d, Q1d)
    for e
        # Calculate offsets to data and temporary arrays
        #<> t_offset = e*tmpSz
        <> u_offset = e*nc*elemsize
        <> v_offset = e*QnD*nc*(dim+2)
        <> d_u_offset = if(transpose, v_offset, u_offset)
        <> d_v_offset = if(transpose, u_offset, v_offset)
        #<> wxs_os = if(d==0,     d_u_offset, t_offset)
        #<> rxs_os = if(d==dim-1, d_v_offset, t_offset) 
 
        #<> pre = ndof
        #<> post = 1 
        #for d
        #    pre = pre*P   
        #end

        with {id_prefix=d_loop}
        for p,d
            #<> Add = (transpose & (d == dim - 1))
            #Could also use prefilled array, may be more efficient
            <> pre = ndof*(P**(dim-1-d))
            <> post = Q**d 

            for a,b,c,j
                <> indw = ((a*Q+j)*post + c) 
                <> indr = ((a*P+b)*post + c) 
                <> rxs = indr + d_v_offset
                <> wxs = indw + d_u_offset

                <> tstride0 = if(transpose, 1, P)
                <> tstride1 = if(transpose, Q, 1)

                if p == d
                    if d == 0
                        if d == dim - 1
                            d_v[wxs] = transpose*d_v[wxs] + grad1d[j*stride0 + b*stride1] * d_u[rxs] 
                        else
                            tmp1[indw] = grad1d[j*stride0 + b*stride1] * d_u[rxs]                 
                        end
                    elif d == dim - 1
                        if d%2 == 0
                            d_v[wxs] = transpose*d_v[wxs] + grad1d[j*stride0 + b*stride1] * tmp1[indr]
                        else    
                            d_v[wxs] = transpose*d_v[wxs] + grad1d[j*stride0 + b*stride1] * tmp0[indr]
                        end
                    elif d%2 == 0
                        tmp0[indw] = grad1d[j*stride0 + b*stride1] * tmp1[indr]
                    else
                        tmp1[indw] = grad1d[j*stride0 + b*stride1] * tmp0[indr]
                    end                         
                else
                    if d == 0
                        if d == dim - 1
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * d_u[rxs] 
                        else
                            tmp1[indw] = interp1d[j*stride0 + b*stride1] * d_u[rxs]                 
                        end
                    elif d == dim - 1
                        if d%2 == 0
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp1[indr]
                        else    
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp0[indr]
                        end
                    elif d%2 == 0
                        tmp0[indw] = interp1d[j*stride0 + b*stride1] * tmp1[indr]
                    else
                        tmp1[indw] = interp1d[j*stride0 + b*stride1] * tmp0[indr]
                    end
                end
            end

         #   pre = pre / P
         #   post = post * Q

        end
        end

        with {dep=d_loop*}
        if transpose
            d_u_offset = d_u_offset + nqpt
        else
            d_v_offset = d_v_offset + nqpt
        end
        end
    end
    """,
    name="kGrad",
    target=lp.OpenCLTarget(),
    assumptions="nelem>0 and dim>0 and pre>0 and post>0 and P>0 and Q>0")
print(kGrad)



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

kWeight = lp.make_kernel(
    ["{ [e]: 0<=e<nelem}",
#     "{ [i]: 0<=i<pre }",
#     "{ [k]: 0<=k<post }"
     "{ [d]: 0<=d<dim }",
#    "{ [dInd]: 2<=dInd<dim+2}",
#    "{ [kInd]: kInd=k+v_shift}",
     "{ [i,j,k]: 0<=i,j,k<Q }" ],
    """
    <> v_shift = (QnD*nc + QnD*nc*dim)
    for e,i,j,k
        <> v_offset = e*QnD*nc*(dim + 2) + v_shift
        <> xs =((i*Q + j)*Q + k) + v_offset
        
        #Above is equivalent to below?
        #<> post = pow(Q, d)
        #<> pre = pow(Q, dim-d-1) 
        #<> xs =((i*Q + j)*post + k) + v_offset
        if d == 0
            d_v[xs] = qweight1d[j]
        else
            d_v[xs] = qweight1d[j]*d_v[xs]
        end
    end

    #for kInd
       
    #d_v[e,dInd,i,j,kInd] = if(dInd==2, qweight1d[j], qweight1d[j]*d_v[e,dInd,i,j,kInd])
    #Scalars only right now, need to incorpate nc otherwise
    #<> jVal = qweight1d[j]
    #if d == 0
    #    d_v[e,dInd,i,j,kInd] = qweight1d[j] {id=deq0}
    #else
    #    d_v[e,dInd,i,j,kInd] = qweight1d[j]*d_v[e,dInd,i,j,kInd] {id=dneq0}
    #end
    #end
    """, 
    name="kWeight",
    target=lp.OpenCLTarget(),
    assumptions="nelem>0 and dim>0 and Q>0"
    )
#kWeight = lp.prioritize_loops(kWeight,"e,d,i,j,k")
print(kWeight)

kernelList1 = [kZero]
kernelList2 = [kCeedTensorContract]
kernelList3 = [kInterp]
kernelList4 = [kGrad]
kernelList5 = [kWeight]

for k in kernelList1:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {"v": np.float64, "elemsize": np.int32, "nc": np.int32})
    code = lp.generate_code_v2(k).device_code()
    print(code)

for k in kernelList2:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "v": np.float64, 
        "u": np.float64, 
        "t": np.float64, 
        "transpose": np.byte, 
        "Add": np.byte,
        "stride0": np.int32,
        "stride1": np.int32,
        "wxs_os": np.int32,
        "rxs_os": np.int32
        }
    )
    code = lp.generate_code_v2(k).device_code()
    print(code)

for k in kernelList3:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "d_v": np.float64, 
        "d_u": np.float64, 
        "tmp0": np.float64, 
        "tmp1": np.float64,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        #"tmpSz": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32,
        "interp1d": np.float64,
        "transpose": np.byte,
        #"Add": np.byte,
        "stride0": np.int32,
        "stride1": np.int32,
        #"wxs_os": np.int32,
        #"rxs_os": np.int32
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)

for k in kernelList4:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "d_v": np.float64, 
        "d_u": np.float64, 
        "tmp0": np.float64, 
        "tmp1": np.float64,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32,
        "interp1d": np.float64,
        "grad1d": np.float64,
        "transpose": np.byte,
        "stride0": np.int32,
        "stride1": np.int32,
        })
    code = lp.generate_code_v2(k).device_code()
    print(code)

for k in kernelList5:
    k = lp.set_options(k, "write_cl")
    k = lp.add_and_infer_dtypes(k, {
        "d_v": np.float64,
        "qweight1d": np.float64,
        "nc": np.int32,
        "QnD": np.int32
    })
    code = lp.generate_code_v2(k).device_code()
    print(code)
