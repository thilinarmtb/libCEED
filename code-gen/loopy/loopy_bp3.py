import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def generate_diffsetupf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    diffsetupf = lp.make_kernel(
        ["{ [i,j,k]: 0<=i<Q and 0<=j,k<3}",
         "{ [kkk,jjj,lll]: 0<=kkk,lll,jjj<3}" ],
        """
        # For reference
        #    *x = in + iOf7[0],
        #    *J = in + iOf7[1];
        #    *w = in + iOf7[2];
        #    *rho = out + oOf7[0]
        #    *rhs = out + oOf7[1]

        M_PI := 3.14159265358979323846
        iind1 := i + iOf7[1]
        oind1 := i + oOf7[1]
        iind2 := i + iOf7[2]

    
        J(jj,kk) := in[3*Q*jj + Q*kk + iind1]

        for i
            for j,k

                <> m = (j + 1) % 3
                <> n = (j + 2) % 3
                <> o = (k + 1) % 3
                <> p = (k + 2) % 3
 
                <> A[k,j] = J(n,p)*J(m,o) - J(n,o)*J(m,p)
            end
            <> s =  sum(k, J(k,0)*A[0,k])
            w := in[iind2] / s
            for jjj,kkk
                out[3*Q*jjj + Q*kkk + oind1] = w * sum(lll, A[jjj,lll]*A[kkk,lll])
            end
            rho := in[iind2] * s
            out[oind1] = rho * M_PI**2 * (1**2 + 2**2 + 3**2) * \
                            sin(M_PI*(0 + 1*in[iind1 + 0*Q])) * \
                            sin(M_PI*(1 + 2*in[iind1 + 1*Q])) * \
                            sin(M_PI*(2 + 3*in[iind1 + 2*Q])) 

        end 
        """,
        name="diffsetupf",
        assumptions="Q > 0",
        target=target
        )

    diffsetupf = lp.fix_parameters(diffsetupf, **constants)
    #diffsetupf = lp.tag_inames(diffsetupf, [("j", "unr"), ("k", "unr")])

    diffsetupf = lp.add_and_infer_dtypes(diffsetupf, {
        "in": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32,
    })
    

    return diffsetupf

def generate_diffusionf(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    diffusionf = lp.make_kernel(
        "{ [i,j]: 0<=i<Q }",
        """
        #    For Reference
        #    *u = in  + iOf7[0],
        #    *rho = in  + iOf7[1];
        #    *v = out + oOf7[0];

        ug(ii, jj) := in[iOf7[0] + ii + Q*jj]
        rhog(ii,jj) := in[iOf7[1] + ii + Q*jj]

        for i
            out[oOf7[0] + i + Q*0] = rhog(i,0)*ug(i,0) + rhog(i,1)*ug(i,1) + rhog(i,2)*ug(i,2)
            out[oOf7[1] + i + Q*1] = rhog(i,1)*ug(i,0) + rhog(i,3)*ug(i,1) + rhog(i,4)*ug(i,2)  
            out[oOf7[1] + i + Q*2] = rhog(i,2)*ug(i,0) + rhog(i,4)*ug(i,1) + rhog(i,5)*ug(i,2)
        end
        """,
        name="diffusionf",
        assumptions="Q > 0",
        target=target
        )

    diffusionf = lp.add_and_infer_dtypes(diffusionf, {
        "in": fp_format,
        "oOf7": np.int32,
        "iOf7": np.int32,
    })
 
    return diffusionf

diffsetupf = generate_diffsetupf()
code = lp.generate_code_v2(diffsetupf).device_code()
print(diffsetupf)
print()
print(code)
print()

diffusionf = generate_diffusionf()
code = lp.generate_code_v2(diffusionf).device_code()
print(diffsetupf)
print()
print(code)
print()
