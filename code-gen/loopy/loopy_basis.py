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
def generate_kZero(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
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

    kZero = lp.fix_parameters(kZero, **constants)

    kZero = lp.tag_inames(kZero, {"e":"g.1"}) 
    if arch == "AMD_GPU":
        kZero = lp.split_iname(kZero, "i", 64, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    elif arch == "NVIDIA_GPU":
        kZero = lp.split_iname(kZero, "i", 32, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    else:
        kZero = lp.split_iname(kZero, "i", 128, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))

    return kZero


def generate_kInterp3d(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kInterp3d = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = P1d
        <> Q = Q1d
        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset
        stride := j*P + b
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)
 

        for e
           <> d_u_offset = u_offset
           <> d_v_offset = v_offset
     
            for d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d
                for a,b,c,j
                    if d == 0 #Should probably assign different iteration variables for each d value instead
                        tmp1[indw] = interp1d[stride] * d_u[rxs] {id=one}
                    elif d == 1
                        tmp0[indw] = interp1d[stride] * tmp1[indr] {id=two,dep=one}
                    else
                        d_v[wxs] = interp1d[stride] * tmp0[indr] {id=three,dep=two}
                    end
                end
            end
            d_v_offset = d_v_offset + nqpt
        end
        """,
        name = "kInterp3d",
        target=target,
        assumptions="nelem>0 and dim=3 and pre>0 and post>0 and P>0 and Q>0"
    )

    kZero = lp.fix_parameters(kInterp3d, **constants)
   
    kInterp3d = lp.prioritize_loops(kInterp3d, "e,d,a,j,b,c")
    kInterp3d = lp.duplicate_inames(kInterp3d,"a,b,c,j", within="id:two")
    kInterp3d = lp.duplicate_inames(kInterp3d,"a,b,c,j", within="id:three")

    kInterp3d = lp.add_and_infer_dtypes(kInterp3d, {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "interp1d": fp_format,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32
    })
 
        
    return kInterp3d


def generate_kInterp3d_T(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kInterp3d_T = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<dim }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = Q1d
        <> Q = P1d
        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset
        stride := j + b*Q
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)
 

        for e
           <> d_u_offset = v_offset
           <> d_v_offset = u_offset
     
            for d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d
                for a,b,c,j
                    if d==0
                        tmp1[indw] = interp1d[stride] * d_u[rxs] {id=one}
                    elif d==1
                        tmp0[indw] = interp1d[stride] * tmp1[indr] {id=two,dep=one}
                    else
                        d_v[wxs] = d_v[wxs] + interp1d[stride] * tmp0[indr] {id=three,dep=two}
                    end
                end
            end
            d_u_offset = d_u_offset + nqpt
        end
        """,
        name = "kInterp3d_T",
        target=target,
        assumptions="nelem>0 and dim=3 and pre>0 and post>0 and P>0 and Q>0"
    )

    kInterp3d_T = lp.fix_parameters(kInterp3d_T, **constants)

    kInterp3d_T = lp.prioritize_loops(kInterp3d_T, "e,d,a,j,b,c")
    kInterp3d_T = lp.duplicate_inames(kInterp3d_T,"a,b,c,j", within="id:two")
    kInterp3d_T = lp.duplicate_inames(kInterp3d_T,"a,b,c,j", within="id:three")

    kInterp3d_T = lp.add_and_infer_dtypes(kInterp3d_T, {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "interp1d": fp_format,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32
        })
 

    return kInterp3d_T


def generate_kGrad3d(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kGrad3d = lp.make_kernel(
        ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<dim }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = P1d
        <> Q = Q1d

        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset
        stride := j*P + b
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)

        for e
            <> d_u_offset = u_offset
            <> d_v_offset = v_offset

            with {id_prefix=d_loop}
            for p,d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d

                if p == d
                    for a,b,c,j
                        if d==0
                            tmp1[indw] = grad1d[stride] * d_u[rxs] {id=one}
                        elif d==1
                            tmp0[indw] = grad1d[stride] * tmp1[indr] {id=two,dep=one}
                        else
                            d_v[wxs] = grad1d[stride] * tmp0[indr] {id=three,dep=two}
                        end
                    end
                else
                    for a,b,c,j
                        if d==0
                            tmp1[indw] = interp1d[stride] * d_u[rxs] {id=four}
                        elif d==1
                            tmp0[indw] = interp1d[stride] * tmp1[indr] {id=five,dep=four}
                        else
                            d_v[wxs] = interp1d[stride] * tmp0[indr] {id=six,dep=five}
                        end
                    end
                end

            end
            end

            d_u_offset = d_u_offset + nqpt
        end
        """,
        name="kGrad3d",
        target=target,
        assumptions="nelem>0 and dim=3 and pre>0 and post>0 and P>0 and Q>0"
    )


    kGrad3d = lp.fix_parameters(kGrad3d, **constants)

    kGrad3d = lp.prioritize_loops(kGrad3d, "e,p,d,a,j,b,c")
    kGrad3d = lp.duplicate_inames(kGrad3d,"a,b,c,j", within="id:two")
    kGrad3d = lp.duplicate_inames(kGrad3d,"a,b,c,j", within="id:three")
    kGrad3d = lp.duplicate_inames(kGrad3d,"a,b,c,j", within="id:four")
    kGrad3d = lp.duplicate_inames(kGrad3d,"a,b,c,j", within="id:five")
    kGrad3d = lp.duplicate_inames(kGrad3d,"a,b,c,j", within="id:six")

    kGrad3d = lp.add_and_infer_dtypes(kGrad3d, {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "interp1d": fp_format,
        "grad1d": fp_format,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32
        })
 

    return kGrad3d


def generate_kGrad3d_T(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kGrad3d_T = lp.make_kernel(
        ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<dim }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = Q1d
        <> Q = P1d

        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset
        stride := j + b*Q
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)

        for e
            <> d_u_offset = v_offset
            <> d_v_offset = u_offset

            with {id_prefix=d_loop}
            for p,d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d

                if p == d
                    for a,b,c,j
                        if d==0
                            tmp1[indw] = grad1d[stride] * d_u[rxs] {id=one}
                        elif d==1
                            tmp0[indw] = grad1d[stride] * tmp1[indr] {id=two,dep=one}
                        else
                            d_v[wxs] = d_v[wxs] + grad1d[stride] * tmp0[indr] {id=three,dep=two}
                        end 
                    end
                else
                    for a,b,c,j
                        if d==0
                            tmp1[indw] = interp1d[stride] * d_u[rxs] {id=four}
                        elif d==1
                            tmp0[indw] = interp1d[stride] * tmp1[indr] {id=five,dep=four}
                        else
                            d_v[wxs] = d_v[wxs] + interp1d[stride] * tmp0[indr] {id=six,dep=five}
                        end
                    end
                end

            end
            end

            d_u_offset = d_v_offset + nqpt
        end
        """,
        name="kGrad3d_T",
        target=target,
        assumptions="nelem>0 and dim=3 and pre>0 and post>0 and P>0 and Q>0"
    )

    kGrad3d_T = lp.fix_parameters(kGrad3d_T, **constants)

    kGrad3d_T = lp.prioritize_loops(kGrad3d_T, "e,p,d,a,j,b,c")
    kGrad3d_T = lp.duplicate_inames(kGrad3d_T,"a,b,c,j", within="id:two")
    kGrad3d_T = lp.duplicate_inames(kGrad3d_T,"a,b,c,j", within="id:three")
    kGrad3d_T = lp.duplicate_inames(kGrad3d_T,"a,b,c,j", within="id:four")
    kGrad3d_T = lp.duplicate_inames(kGrad3d_T,"a,b,c,j", within="id:five")
    kGrad3d_T = lp.duplicate_inames(kGrad3d_T,"a,b,c,j", within="id:six")

    kGrad3d_T = lp.add_and_infer_dtypes(kGrad3d_T, {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "interp1d": fp_format,
        "grad1d": fp_format,
        "elemsize": np.int32,
        "ndof": np.int32,
        "QnD": np.int32,
        "Q1d": np.int32,
        "nc": np.int32,
        "P1d": np.int32,
        "nqpt": np.int32
        })

    return kGrad3d_T


# Only works for 3D, if need 2D or 1D add separate cases to handle
def generate_kWeight(constants={},arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
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
            <> qw_i = qw(i)
            <> qw_j = qw(j)
            <> qw_k = qw(k)
            d_v[xs1] = qw_i*qw_j*qw_k*d_v[xs1]**2

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
        target=target,
        assumptions="nelem>0 and dim>0 and Q>0"
        )

    kWeight = lp.add_and_infer_dtypes(kWeight, {
        "d_v": fp_format,
        "qweight1d": fp_format,
        "nc": np.int32,
        "QnD": np.int32
    })

        #if d == 0

    kWeight = lp.fix_parameters(kWeight, **constants)

    #kWeight = lp.precompute(kWeight, "v_shift")
    #kWeight = lp.precompute(kWeight, "m")
    #kWeight = lp.precompute(kWeight, "QQ")
    #kWeight = lp.precompute(kWeight, "v_offset")
    #kWeight = lp.precompute(kWeight, "qw_i")
    kWeight = lp.prioritize_loops(kWeight,"e,i,j,k")
    #kWeight = lp.add_prefetch(kWeight, "qweight1d", "i,j,k")

    return kWeight

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
    
generators = [generate_kZero, 
              generate_kInterp3d,
              generate_kInterp3d_T,
              generate_kGrad3d,
              generate_kGrad3d_T,
              generate_kWeight]

for generator in generators:
    k = generator()
    #print(k)
    code = lp.generate_code_v2(k).device_code()
    print(code)
    print()


