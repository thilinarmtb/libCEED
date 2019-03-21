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

'''
def generate_kInterp(arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kInterp = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = if(transpose, Q1d, P1d)
        <> Q = if(transpose, P1d, Q1d)
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)
        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset
        stride0 := j + b*Q #if(transpose, 1, P)
        stride1 := j*P + b #if(transpose, Q, 1)

        for e
            <> d_u_offset = if(transpose, v_offset, u_offset)
            <> d_v_offset = if(transpose, u_offset, v_offset)
 
            with {id_prefix=d_loop}
            for d
                #<> Add = (transpose & (d == dim - 1))
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d 

                for a,b,c,j
                    if transpose
                        if d == 0
                            if d == dim - 1
                                d_v[wxs] = d_v[wxs] + interp1d[stride0] * d_u[rxs]  
                            else
                                tmp1[indw] = interp1d[stride0] * d_u[rxs]                 
                            end
                        elif d == dim - 1
                            if d%2 == 0
                                d_v[wxs] = d_v[wxs] + interp1d[stride0] * tmp1[indr]
                            else    
                                d_v[wxs] = d_v[wxs] + interp1d[stride0] * tmp0[indr]
                            end
                        elif d%2 == 0
                            tmp0[indw] = interp1d[stride0] * tmp1[indr]
                        else
                            tmp1[indw] = interp1d[stride0] * tmp0[indr]
                        end
                    else
                        if d == 0
                            if d == dim - 1
                                d_v[wxs] = interp1d[stride1] * d_u[rxs] 
                            else
                                tmp1[indw] = interp1d[stride1] * d_u[rxs]                 
                            end
                        elif d == dim - 1
                            if d%2 == 0
                                d_v[wxs] = interp1d[stride1] * tmp1[indr]
                            else    
                                d_v[wxs] = interp1d[stride1] * tmp0[indr]
                            end
                        elif d%2 == 0
                            tmp0[indw] = interp1d[stride1] * tmp1[indr]
                        else
                            tmp1[indw] = interp1d[stride1] * tmp0[indr]
                        end
                    end
                end
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

    return kInterp
   
print(generate_kInterp())
'''

'''
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
'''

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

'''
kZero = generate_kZero()
kernelList1 = [kZero]
kCeedTensorContract = generate_kCeedTensorContract()
kernelList2 = [kCeedTensorContract]
kernelList3 = [kInterp]
kernelList4 = [kGrad]
kWeight = generate_kWeight()
kernelList5 = [kWeight]
'''
'''
for k in kernelList1:
    print(k)
    print()
    code = lp.generate_code_v2(k).device_code()
    print(code)
'''
'''
for k in kernelList2:
    print(k)
    print()
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
'''
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
'''
'''
for k in kernelList5:
    print(k)
    print()
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()
'''
