import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

def generate_kInterp(transpose=False):

    loopyCode = ""

    if transpose:
        loopyCode += """
                     P := Q1D
                     Q := P1D
                     stride0 := 1
                     stride1 := P1D
                     u_stride := nqpt
                     v_stride := ncomp*elemsize
                     u_comp_stride := nelem*nqpt
                     v_comp_stride := elemsize
                     u_size := nqpt
                     """
    else:
        loopyCode += """
                     P := P1D
                     Q := Q1D
                     stride0 := P1D
                     stride1 := 1
                     u_stride := ncomp*elemsize
                     v_stride := nqpt
                     u_comp_stride := elemsize
                     v_comp_stride := nelem * nqpt
                     u_size := elemsize
                     """

    loopyCode += """
                 pre(d) := u_size*(P**(dim-1-d))
                 post(d) := Q**d 
                 c(d,k) := k % post(d)
                 j(d,k) := (k / post(d)) % Q
                 a(d,k) := k / (post(d) * Q)
                 u_offset := elem*u_stride + comp*u_comp_stride
                 v_offset := elem*v_stride + comp*v_comp_stride
                 in_offset(d) := if(d % 2, BASIS_BUF_LEN, 0) # Can precalculate these
                 out_offset(d) := if(d % 2, 0, BASIS_BUF_LEN)

                 <> PP = P
                 #<> uu_size = u_size
                 for elem,comp
                     #Probalamente se podria encontrar una manera de hacer sin copiar pero el GPU lo requiere de todos modos
                     #<> tmp[kk] = u[u_offset + kk] 
                     for d #Esto deberia iterar a menos de dim-1
                         <> writeLen = pre(d) * post(d) * Q
                         for k
                            if d == 0 
                                <> tmp2[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * u[u_offset + (a(d,k)*P + b)*post(d) + c(d,k)])
                            elif d%2 == 0
                                # This probably will not work because loading from and saving to same array
                                # tmp[out_offset(d) + k] = 0#sum(b, interp1d[j(d,k)*stride0 + b*stride1]) #* tmp[in_offset(d) + (a(d,k)*P + b)*post(d) + c(d,k)])
                                tmp2[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * tmp[(a(d,k)*P + b)*post(d) + c(d,k)])
                            else 
                                tmp[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * tmp2[(a(d,k)*P + b)*post(d) + c(d,k)])
                            end
                         end
                     end
                     <> writeLen2 = pre(dim-1) * post(dim-1) * Q
                     for kkk # Note: Change this iteration variable name at some point
                        #v[v_offset + kkk] = sum(b, interp1d[j(dim-1,kkk)*stride0 + b*stride1] * tmp[in_offset(dim-1) + (a(dim-1,kkk)*P + b)*post(dim-1) + c(dim-1,kkk)])
                        if dim - 1 == 0 
                            v[v_offset + kkk] = sum(b, interp1d[j(dim-1,kkk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kkk)*P + b)*post(dim-1) + c(dim-1,kkk)])
                        elif dim-1 % 2 == 0
                            v[v_offset + kkk] = sum(b, interp1d[j(dim-1,kkk)*stride0 + b*stride1] * tmp[(a(dim-1,kkk)*P + b)*post(dim-1) + c(dim-1,kkk)])
                        else
                            v[v_offset + kkk] = sum(b, interp1d[j(dim-1,kkk)*stride0 + b*stride1] * tmp2[(a(dim-1,kkk)*P + b)*post(dim-1) + c(dim-1,kkk)])
                        end
                     end
                 end
                 """

    kInterp = lp.make_kernel(
        ["{ [elem]: 0<=elem<nelem }",
         "{ [comp]: 0<=comp<ncomp }",
         "{ [d]: 0<=d<dim-1 }",
         #"{ [kk]: 0<=kk<uu_size}",
         "{ [k]: 0<=k<writeLen}",
         "{ [kkk]: 0<=kkk<writeLen2}",
         "{ [b]: 0<=b<PP}"],
        loopyCode,
        name="kWeight"
        #target=target,
        #kernel_data=kernel_data
    )

    print(kInterp)

    #print(loopyCode)

'''
def generate_kInterp()

    loopyCode = ""
                

def head(transpose=False, interleaved=False):
    loopyCode = """
                c := k % post
                j := (k / post) % Q
                a := k / (post * Q)
                """
    if not transpose:
        loopyCode += """
                     P := P1D
                     Q := Q1D
                     stride0 := P1D
                     stride1 := 1
                     u_stride := ncomp*elemsize
                     v_stride := nqpt
                     """
        if interleaved:
            loopyCode += """
                         u_comp_stride := elemsize
                         v_comp_stride := nelem*nqpt
                         u_size := elemsize
                         """
            
    else:
        loopyCode += """
                      P := Q1D
                      Q := P1D
                      stride0 := 1
                      stride1 := P1D
                      u_stride := nqpt
                      v_stride := ncomp*elemsize
                      """
         if interleaved:
            loopyCode += """
                         u_comp_stride := nelem*nqpt
                         v_comp_stride := elemsize
                         u_size := nqpt
                         """
    if interleaved:
        loopyCode += """
                     u_offset := 
                     preInit := u_stride 
                     """
    else:
        loopyCode += """
                     preInit := u_size
                     """    

"""    
for elem, comp
    <> pre = preInit
    <> post = 1
    for d
        pre = pre / P
        post = 
        <> outOffset = 
        <> inOffset = 
        writeLen = pre*post*Q
        for k
            out[k] = sum(b, interp1d[j*stride0 + b*stride1] * in[]
        end
        post = post*Q
    end
end

def generate_kZero(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data=["v"]
    dtypes={"v": fp_format}
    if constants=={}:
        kernel_data += ["elemsize", "nc", "nelem"]
        dtypes.update({"elemsize": np.int32, "nc": np.int32})

    kZero = lp.make_kernel(
        "{ [e,i,j]: 0<=e<nelem and 0<=i<elemsize and 0<=j<nc }",
        """
        v[e,i,j] = 0
        """,
        name="kZero",
        assumptions="nelem > 0 and nc > 0 and elemsize > 0",
        kernel_data=kernel_data,
        target=target
    )

    """
    kZero = lp.tag_inames(kZero, {"e":"g.1"}) 
    if arch == "AMD_GPU":
        workgroup_size=64
    elif arch == "NVIDIA_GPU":
        workgroup_size=32
    else:
        workgroup_size=128
    kZero = lp.split_iname(kZero, "i", workgroup_size, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    """

    kZero = lp.fix_parameters(kZero, **constants)
    kZero = lp.add_and_infer_dtypes(kZero, dtypes)

    return kZero


def generate_kInterp3d_(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = [
        "QnD", "transpose", "tmode", "tmp0", "tmp1", "interp1d", "d_u", "d_v" ]
    if constants=={}:
        kernel_data = kernel_data + [
            "elemsize","nc","ndof","nelem", "nqpt", "P1d", "Q1d", "tmpSz"]

    kInterp3d_ = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<3 }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        dim := 3
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
        name = "kInterp3d_",
        target=target,
        assumptions="nelem>0 and pre>0 and post>0 and P>0 and Q>0",
        kernel_data=kernel_data
    )

    kInterp3d_ = lp.fix_parameters(kInterp3d_, **constants)
   
    kInterp3d_ = lp.prioritize_loops(kInterp3d_, "e,d,a,j,b,c")
    kInterp3d_ = lp.duplicate_inames(kInterp3d_,"a,b,c,j", within="id:two")
    kInterp3d_ = lp.duplicate_inames(kInterp3d_,"a,b,c,j", within="id:three")

    kInterp3d_ = lp.add_and_infer_dtypes(kInterp3d_, {
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
        "nqpt": np.int32,

        "tmode": np.int32,
        "transpose": np.int32,
        "tmpSz": np.int32
    })
 
        
    return kInterp3d_


def generate_kInterp3d_T(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = [
        "QnD", "transpose", "tmode", "tmp0", "tmp1", "interp1d", "d_u", "d_v" ]
    if constants=={}:
        kernel_data = kernel_data + [
            "elemsize","nc","ndof","nelem", "nqpt", "P1d", "Q1d", "tmpSz"]

    kInterp3d_T = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<3 }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = Q1d
        <> Q = P1d
        dim := 3
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
        assumptions="nelem>0 and pre>0 and post>0 and P>0 and Q>0"
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

def generate_kInterp(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = [
        "QnD", "transpose", "tmode", "tmp0", "tmp1", "interp1d", "d_u", "d_v" ]
    dtypes = {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "interp1d": fp_format,
        "transpose": np.int32,
        "QnD": np.int32,
        "tmode": np.int32,
    }

    if constants=={}:
        kernel_data += [
            "dim", "elemsize","nc","ndof","nelem", "nqpt", "P1d", "Q1d", "tmpSz"]
        dtypes.update({
            "elemsize": np.int32,
            "ndof": np.int32,
            "Q1d": np.int32,
            "nc": np.int32,
            "P1d": np.int32,
            "nqpt": np.int32,
            "nelem": np.int32,
            "tmpSz": np.int32,
            "dim": np.int32
         })

    kInterp = lp.make_kernel(
        ["{ [e,d]: 0<=e<nelem and 0<=d<dim}",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        u_offset := e*nc*elemsize
        v_offset := e*QnD*nc*(dim+2)
        indw := ((a*Q+j)*post + c) 
        indr := ((a*P+b)*post + c) 
        rxs := indr + d_v_offset
        wxs := indw + d_u_offset

        <> P = if(transpose, Q1d, P1d)
        <> Q = if(transpose, P1d, Q1d)
        for e
            <> d_u_offset = if(transpose, v_offset, u_offset)
            <> d_v_offset = if(transpose, u_offset, v_offset)
            
            with {id_prefix=d_loop}
            for d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d 

                for a,b,c,j
                    <> stride0 = if(transpose, 1, P)
                    <> stride1 = if(transpose, Q, 1)

                    if d == 0
                        if d == dim - 1
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * d_u[rxs] 
                        else
                            tmp1[indw] = interp1d[j*stride0 + b*stride1] * d_u[rxs]                 
                        end
                    elif d == dim - 1
                        if d%2 == 0
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp0[indr]
                        else    
                            d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp1[indr]
                        end
                    elif d%2 == 0
                        tmp1[indw] = interp1d[j*stride0 + b*stride1] * tmp0[indr]
                    else
                        tmp0[indw] = interp1d[j*stride0 + b*stride1] * tmp1[indr]
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
        target=target,
        assumptions="nelem>0 and pre>0 and post>0 and P>0 and Q>0",
        kernel_data=kernel_data
    )

    kInterp = lp.prioritize_loops(kInterp, "e,d,a,j,b,c")

    kInterp = lp.fix_parameters(kInterp, **constants)
    kInterp = lp.add_and_infer_dtypes(kInterp,dtypes)

    return kInterp
 

def generate_kGrad3d_(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = [
        "QnD", "transpose", "tmode", "tmp0", 
        "tmp1", "grad1d", "interp1d", "d_u", "d_v" ]
    if constants=={}:
        kernel_data = kernel_data + [
            "elemsize","nc","ndof","nelem", "nqpt", "P1d", "Q1d", "tmpSz"]

    kGrad3d_ = lp.make_kernel(
        ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<3 }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = P1d
        <> Q = Q1d

        dim := 3
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
        name="kGrad3d_",
        target=target,
        assumptions="nelem>0 and pre>0 and post>0 and P>0 and Q>0"
    )


    kGrad3d_ = lp.fix_parameters(kGrad3d_, **constants)

    kGrad3d_ = lp.prioritize_loops(kGrad3d_, "e,p,d,a,j,b,c")
    kGrad3d_ = lp.duplicate_inames(kGrad3d_,"a,b,c,j", within="id:two")
    kGrad3d_ = lp.duplicate_inames(kGrad3d_,"a,b,c,j", within="id:three")
    kGrad3d_ = lp.duplicate_inames(kGrad3d_,"a,b,c,j", within="id:four")
    kGrad3d_ = lp.duplicate_inames(kGrad3d_,"a,b,c,j", within="id:five")
    kGrad3d_ = lp.duplicate_inames(kGrad3d_,"a,b,c,j", within="id:six")

    kGrad3d_ = lp.add_and_infer_dtypes(kGrad3d_, {
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
 
    return kGrad3d_


def generate_kGrad3d_T(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kGrad3d_T = lp.make_kernel(
        ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<3 }",
         "{ [a,j,c,b]: 0<=a<pre and 0<=j<Q and 0<=c<post and 0<=b<P }"],
        """
        <> P = Q1d
        <> Q = P1d

        dim := 3
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
        assumptions="nelem>0 and pre>0 and post>0 and P>0 and Q>0"
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

def generate_kGrad(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data = [
        "QnD", "transpose", "tmode", "tmp0", "tmp1", "grad1d", "interp1d", "d_u", "d_v" ]
    dtypes = {
        "d_v": fp_format, 
        "d_u": fp_format, 
        "tmp0": fp_format, 
        "tmp1": fp_format,
        "grad1d": fp_format,
        "interp1d": fp_format,
        "transpose": np.int32,
        "QnD": np.int32,
        "tmode": np.int32,
    }

    if constants=={}:
        kernel_data += [
            "dim", "elemsize","nc","ndof","nelem", "nqpt", "P1d", "Q1d", "tmpSz"]
        dtypes.update({
            "elemsize": np.int32,
            "ndof": np.int32,
            "Q1d": np.int32,
            "nc": np.int32,
            "P1d": np.int32,
            "nqpt": np.int32,
            "nelem": np.int32,
            "tmpSz": np.int32,
            "dim": np.int32
         })
 
    kGrad = lp.make_kernel(
        ["{ [e,d,p]: 0<=e<nelem and 0<=d,p<dim and dim=3}",
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
 
        for e
            <> d_u_offset = if(transpose, v_offset, u_offset)
            <> d_v_offset = if(transpose, u_offset, v_offset)

            with {id_prefix=d_loop}
            for p,d
                <> pre = ndof*(P**(dim-1-d))
                <> post = Q**d 

                for a,b,c,j
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
                                d_v[wxs] = transpose*d_v[wxs] + grad1d[j*stride0 + b*stride1] * tmp0[indr]
                            else    
                                d_v[wxs] = transpose*d_v[wxs] + grad1d[j*stride0 + b*stride1] * tmp1[indr]
                            end
                        elif d%2 == 0
                            tmp1[indw] = grad1d[j*stride0 + b*stride1] * tmp0[indr]
                        else
                            tmp0[indw] = grad1d[j*stride0 + b*stride1] * tmp1[indr]
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
                                d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp0[indr]
                            else    
                                d_v[wxs] = transpose*d_v[wxs] + interp1d[j*stride0 + b*stride1] * tmp1[indr]
                            end
                        elif d%2 == 0
                            tmp1[indw] = interp1d[j*stride0 + b*stride1] * tmp0[indr]
                        else
                            tmp0[indw] = interp1d[j*stride0 + b*stride1] * tmp1[indr]
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
        name="kGrad",
        target=target,
        assumptions="nelem>0 and dim>0 and pre>0 and post>0 and P>0 and Q>0",
        kernel_data=kernel_data
    )

    kGrad = lp.fix_parameters(kGrad, **constants)
    kGrad = lp.add_and_infer_dtypes(kGrad, dtypes)

    return kGrad
'''
def generate_kWeight(constants={},dim=3,arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    kernel_data= [
        lp.GlobalArg("qweight1d", fp_format),
        lp.GlobalArg("w", fp_format)
    ]

    if constants=={}:
        kernel_data += [
            lp.ValueArg("Q1D", np.int32),
            lp.ValueArg("nelem", np.int32)
        ]

    iterVars = [ "{ [e]: 0<=e<nelem}" ]
    loopyCode = ""
    loopPriority="e"  
 
    if dim == 1:
        iterVars += ["{ [i]: 0<=i<Q1D }"] 
        loopyCode = """
                    w[e*Q1D +i] = qweight1d[i]
                    """ 
        loopPriority += ",i"
    elif dim == 2:
        iterVars += ["{ [i,j]: 0<=i,j<Q1D }"]
        loopyCode = """        
                    w[e*Q1D*Q1D + j*Q1D + i] = qweight1d[i]*qweight1d[j]
                    """ 
        loopPriority += ",i,j"
    elif dim == 3:
        iterVars += ["{ [i,j,k]: 0<=i,j,k<Q1D }"]
        loopyCode = """        
                    index := e*Q1D*Q1D*Q1D + k*Q1D*Q1D + j*Q1D + i
                    w[index] = qweight1d[i]*qweight1d[j]*qweight1d[k]
                    """ 
        loopPriority += ",i,j,k"
    else:
       raise Exception("Unknown dimension specified in generate_kWeight()")

    kWeight = lp.make_kernel(
        iterVars,
        loopyCode,
        name="kWeight",
        target=target,
        assumptions="nelem>0 and Q1D>0",
        kernel_data=kernel_data
    )

    kWeight = lp.prioritize_loops(kWeight,loopPriority)
    kWeight = lp.fix_parameters(kWeight, **constants)


    print(kWeight)
    code = lp.generate_code_v2(kWeight).device_code()
    print(code)

    return kWeight
#generate_kWeight(dim=3)
generate_kInterp()