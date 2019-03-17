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

def generate_kRestrict():
    kRestrict = lp.make_kernel(
        [ "{ [e,j,k,d]: 0<=e<nblk and 0<=j<blksize and 0<=k<elemsize and 0<=d<ncomp}",
          "{ [jj]: 0<=jj<jmax }"
            ],
        """
        val1 := e*blksize + j
        val2 := nelem - 1
        val3 := nelem - e

        out0 := e*elemsize*ncomp*blksize + k*ncomp*blksize + d*blksize + j 
        in0 := e*blksize*ncomp*elemsize + j*ncomp*elemsize + k*ncomp + d
        in1 := val2*ncomp*elemsize + k*ncomp + d

        out1 := e*blksize*elemsize*ncomp + d*blksize*elemsize + k*blksize + j
        in2 := indices[e*blksize*elemsize + j*elemsize + k] + ndof*d
        in3 := indices[e*blksize*elemsize + j*elemsize + k]*ncomp + d


        in4 := e*elemsize*ncomp*blksize + k*ncomp*blksize + blksize*d + jj
        out2 := e*elemsize*ncomp*blksize + jj*ncomp*elemsize + k*ncomp + d

        in5 := e*ncomp*blksize*elemsize + d*blksize*elemsize + k*blksize + jj
        out3 := indices[e*elemsize*blksize + k*blksize + jj] + ndof*d 
        out4 := indices[e*elemsize*blksize + k*blksize + jj]*ncomp + d


        for e
            if tmode == CEED_NOTRANSPOSE
                for k,d,j
                    if not use_ind
                        # No indices provided, Identity Restriction
                        if val1 < val2
                            vv[out0] = uu[in0] {id=case0}
                        else
                            vv[out0] = uu[in1] {id=case1}
                        end
                    else
                        if lmode == CEED_NOTRANSPOSE
                            vv[out1] = uu[in2] {id=case2}
                        else
                            vv[out1] = uu[in3] {id=case3}
                        end
                    end
                end
            else
                <> jmax = if(blksize < val3, blksize, val3) {id=jmax}
                for k, d, jj
                    if not use_ind
                        # Restriction from evector to lvector
                        # Performing v += r^T * u
                        # No indices provided, Identity Restriction

                        vv[out2] = vv[out2] + uu[in4] {id=case4, dep=jmax}
                    else
                        # Indices provided, standard or blocked restriction
                        # uu has shape [elemsize, ncomp, nelem]
                        # vv has shape [ndof, ncomp] 
                        # Iteration bound set to discard padding elements

                        if not lmode == CEED_NOTRANSPOSE
                            vv[out3] = vv[out3] + uu[in5] {id=case5, dep=jmax}
                        else
                            vv[out4] = vv[out4] + uu[in5] {id=case6, dep=jmax}
                        end 
                    end
                end
            end
        end
        """,
        name = "kRestrict",
        assumptions="blksize > 0 and elemsize > 0 and ncomp > 0 and nblk > 0",
        target=lp.OpenCLTarget()
    )
    #kRestrict = lp.duplicate_inames(kRestrict, "e,k,d,j", within="id:case0")
    #kRestrict = lp.duplicate_inames(kRestrict, "e,k,d,j", within="id:case1")
    #kRestrict = lp.duplicate_inames(kRestrict, "e,k,d,j", within="id:case2")
    #kRestrict = lp.duplicate_inames(kRestrict, "e,k,d,j", within="id:case3")
    #kRestrict = lp.duplicate_inames(kRestrict, "k,d,jj", within="id:case4")
    #kRestrict = lp.duplicate_inames(kRestrict, "k,d,jj", within="id:case5")
    #kRestrict = lp.duplicate_inames(kRestrict, "k,d,jj", within="id:case6")

    kRestrict = lp.add_and_infer_dtypes(kRestrict, {
        "uu": np.float64,
        "vv": np.float64,
        "indices": np.int32,
        "nelem": np.int32,
        "tmode": np.int32,
        "use_ind": np.int32,
        "lmode": np.int32,
        "ndof": np.int32,
        "CEED_NOTRANSPOSE": np.int32})

    return kRestrict

kRestrict = generate_kRestrict()
print(kRestrict)
print()
code = lp.generate_code_v2(kRestrict).device_code()
print(code)
