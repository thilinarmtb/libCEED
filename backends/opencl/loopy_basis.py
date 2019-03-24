import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

# setup
# -----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

# ------
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

    '''
    kZero = lp.tag_inames(kZero, {"e":"g.1"}) 
    if arch == "AMD_GPU":
        workgroup_size=64
    elif arch == "NVIDIA_GPU":
        workgroup_size=32
    else:
        workgroup_size=128
    kZero = lp.split_iname(kZero, "i", workgroup_size, inner_tag="l.0",outer_tag="g.0", slabs=(0,1))
    '''

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

# Only works for 3D, if need 2D or 1D add separate cases to handle
def generate_kWeight(constants={},arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    
    kernel_data= ["QnD", "Q", "qweight1d", "d_v"]
    dtypes = {
        "d_v": fp_format,
        "qweight1d": fp_format,
        "QnD": np.int32
    }

    if constants=={}:
        kernel_data += ["dim", "nc", "nelem"]
        dtypes.update({"dim": np.int32, "nc": np.int32, "nelem": np.int32 })

    kWeight = lp.make_kernel(
        ["{ [e]: 0<=e<nelem}",
         "{ [d]: 0<=d<dim }",
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
            qw_i := qw(i)
            qw_j := qw(j)
            qw_k := qw(k)
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
        assumptions="nelem>0 and dim>0 and Q>0",
        kernel_data=kernel_data
        )

    kWeight = lp.prioritize_loops(kWeight,"e,i,j,k")

    kWeight = lp.add_and_infer_dtypes(kWeight, dtypes)
    kWeight = lp.fix_parameters(kWeight, **constants)


    return kWeight

arg_len = len(sys.argv)
if arg_len != 4:
    print("Usage: python loopy_basis.py kernel_name arch '{\"c1\": val1, ... }'")
    print("Example: python loopy_basis.py kZero '{\"elemsize\": 8, ... }'")
    sys.exit(1)

kernel_name = sys.argv[1]
arch = sys.argv[2]
constants = json.loads(sys.argv[3])

if kernel_name == 'kZero':
    k = generate_kZero(constants, arch)
elif kerel_name == 'kInterp':
    k = generate_kInterp(constants, arch)
elif kerel_name == 'kGrad':
    k = generate_kGrad(constants, arch)
elif kerel_name == 'kWeight':
    k = generate_kWeight(constants, arch)
else:
    print("Invalid kernel name: {}".format(kernel_name))
    sys.exit(1)

code = lp.generate_code_v2(k).device_code()
print(code)
print()
