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

# ------
def generate_kZero(arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kZero = lp.make_kernel(
        "{ [e,i]: 0<=e<nelem and 0<=i<vsize }",
        """
        m := nc*elemsize
        v[e*m + i] = 0
        """,
        name="kZero",
        assumptions="nelem > 0 and vsize > 0",
        target=target
    )
    kZero = lp.add_and_infer_dtypes(kZero, {"v": fp_format, "elemsize": np.int32, "nc": np.int32})

    kZero = lp.tag_inames(kZero, {"e":"g.1"}) 
    if arch == "AMD_GPU":
        kZero = lp.split_iname(kZero, "i", 64, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    elif arch == "NVIDIA_GPU":
        kZero = lp.split_iname(kZero, "i", 32, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    else:
        kZero = lp.split_iname(kZero, "i", 128, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))

    return kZero

def generate_kCeedTensorContract(arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kCeedTensorContract = lp.make_kernel(
        ["{ [a,b,c,j]: 0<=a<A and 0<=b<B and 0<=c<C and 0<=j<J }"],
        """
        wxs := ((a*J+j)*C+c) + wxs_os
        rxs := ((a*B+b)*C+c) + rxs_os
         
        if Add
            if transpose
                v[wxs] = v[wxs] + t[j + b*J] * u[rxs] {id=Add_t}
            else
                v[wxs] = v[wxs] + t[j*B + b] * u[rxs] {id=Add}
            end
        else
            if transpose
                v[wxs] = t[j + b*J] * u[rxs] {id=no_Add_t}
            else 
                v[wxs] = t[j*B + b] * u[rxs] {id=no_Add}
            end
        end 
        """,
        name = "kCeedTensorContract",
        target=target,
        assumptions="A>0 and B>0 and C>0 and J>0"
        )

    kCeedTensorContract = lp.prioritize_loops(kCeedTensorContract, "a,j,b,c")
    kCeedTensorContract = lp.duplicate_inames(kCeedTensorContract,"a,b,c,j", within="id:no_Add")
    kCeedTensorContract = lp.duplicate_inames(kCeedTensorContract,"a,b,c,j", within="id:no_Add_t")
    kCeedTensorContract = lp.duplicate_inames(kCeedTensorContract,"a,b,c,j", within="id:Add")

    # Figure out some additional optimizations here 

    kCeedTensorContract = lp.add_and_infer_dtypes(kCeedTensorContract, {
        "v": fp_format, 
        "u": fp_format, 
        "t": fp_format, 
        "transpose": np.byte, 
        "Add": np.byte,
        "wxs_os": np.int32,
        "rxs_os": np.int32
        }
    )
    return kCeedTensorContract
 

#print(kCeedTensorContract)


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


#print(kInterp)

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
#print(kGrad)



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

# Only works for 3D, if need 2D or 1D add separate cases to handle
kWeight = lp.make_kernel(
    ["{ [e]: 0<=e<nelem}",
#     "{ [ii]: 0<=ii<QQ }",
     "{ [d]: 0<=d<dim and dim=3 }",
     "{ [i,j,k]: 0<=i,j,k<Q }" ],
    """
    v_shift := (QnD*nc + QnD*nc*dim)
    m := QnD*nc*(dim + 2) 
    v_offset := e*m + v_shift
    QQ := Q*Q
    ii := Q*i + k
    xs0 := ((ii * Q + j)    + 0 ) + v_offset
    #xs1 := ((i  * Q + j)*Q  + k ) + v_offset
    xs1 := i*QQ + j*Q + k + v_offset
    xs2 := ((0  * Q + j)*Q*Q + ii) + v_offset
    qw(s) := qweight1d[s]
    #xs := ((i*Q**(dim-d-1) + j)*Q + k) + v_offset
    
    #post := Q**d
    #pre := Q**(dim-d-1)
    #xs := ((i*Q + j)*post + k) + v_offset
    #<> val = pre #For some reason won't generate val without this
    for e,i,j,k
        <> ind = xs1
        d_v[ind] = qw(i)*qw(j)*qw(k)*d_v[ind]**2
        # For contiguous accesses need separate loops with separate orderings?
        #Race condition if do this? 
        #Ideal way to do this is probably coordinate so all values
        # local memory.
        #d_v[xs0] = qweight1d[j] {id=d0} #Possibly tag each for separate loops #i,k,j
        #with {id_prefix=d1}
        #    <> ind1 = xs1
        #    d_v[ind1] = qweight1d[j]*d_v[ind1] #i,j,k
        #end
        #with {id_prefix=d2}
        #    <> ind2 = xs2
        #    d_v[ind2] = qweight1d[j]*d_v[ind2]#j,i,k
        #end
        #Above xs is equivalent to below?
        #<> post = Q**d
        #<> pre = Q**(dim-d-1) 
        #if i < pre and k < post
        #    if d == 0
        #        d_v[xs] = qweight1d[j]
        #    else    
        #        d_v[xs] = qweight1d[j]*d_v[xs]
        #    end
        #end
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

kWeight = lp.precompute(kWeight, "v_shift")
kWeight = lp.precompute(kWeight, "m")
kWeight = lp.precompute(kWeight, "QQ")
kWeight = lp.prioritize_loops(kWeight,"e,i,j,k")
kWeight = lp.add_prefetch(kWeight, "qweight1d", "i,j,k")
kWeight = lp.precompute(kWeight, "v_offset")
#kWeight = lp.duplicate_inames(kWeight,"i,j,k", within="id:d0")
#kWeight = lp.duplicate_inames(kWeight,"i,j,k", within="id:d1*")
#kWeight = lp.duplicate_inames(kWeight,"i,j,k", within="id:d2*")
#kWeight = lp.prioritize_loops(kWeight, "i_0,k_0,j_0")
#kWeight = lp.prioritize_loops(kWeight, "i_1,j_1,k_1")
#kWeight = lp.prioritize_loops(kWeight, "j_2,i_2,k_2")
#kWeight = lp.tag_inames(kWeight, [("e", "g.0")])
#kWeight = lp.precompute(kWeight, "xs2(ii,j)")
#kWeight = lp.tag_inames(kWeight, [("d", "unr")])
#kWeight = lp.precompute(kWeight, "post", "d")
#kWeight = lp.precompute(kWeight, "pre", "d")
#print(kWeight)

kZero = generate_kZero()
kernelList1 = [kZero]
kCeedTensorContract = generate_kCeedTensorContract()
kernelList2 = [kCeedTensorContract]
kernelList3 = [kInterp]
kernelList4 = [kGrad]
kernelList5 = [kWeight]

'''
for k in kernelList1:
    print(k)
    print()
    code = lp.generate_code_v2(k).device_code()
    print(code)
'''

for k in kernelList2:
    print(k)
    print()
    code = lp.generate_code_v2(k).device_code()
    print(code)

'''
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
'''
