import numpy as np
import loopy as lp
try: from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2
except Exception: print("loopy.version not imported")
import sys
import json

TRANSPOSE = 1
INTERLEAVE = 2

def generate_kInterp(constants={},version=0, arch="INTEL_CPU", target=lp.OpenCLTarget(), fp_format=np.float64):

    kernel_data = [
        lp.GlobalArg("interp1d", fp_format,shape=lp.auto),
        lp.GlobalArg("u", fp_format, shape=lp.auto),
        lp.GlobalArg("v", fp_format, shape=lp.auto),
        #lp.GlobalArg("tmp", fp_format, shape=None),
        #lp.GlobalArg("tmp2", fp_format, shape=None),
    ]
    if not "nelem" in constants:
        kernel_data += [lp.ValueArg("nelem", np.int32)]

    mandatory_constants = ["elemsize", "ncomp", "nqpt", "dim", "P1D", "Q1D"]
    for c in mandatory_constants:
        if not c in constants:
            print("Please specify " + c)
            #kernel_data += [lp.ValueArg(c, np.int32)]

    # Remove this when finished testing
    if not "dim" in constants:
        constants["dim"] = 3

    loopyCode = ""

    elemsize  = constants["elemsize"]
    ncomp = constants["ncomp"]
    nqpt = constants["nqpt"]
    dim = constants["dim"]
    P1D = constants["P1D"]
    Q1D = constants["P1D"]   
    

    if not version & TRANSPOSE: #Not transpose
        loopyCode += """
                     P := P1D
                     Q := Q1D
                     stride0 := P1D
                     stride1 := 1
                     """
        P = P1D
        Q = Q1D
        stride0 = P1D
        stride1 = 1
    else: #Transpose
        loopyCode += """
                     P := Q1D
                     Q := P1D
                     stride0 := 1
                     stride1 := P1D
                     """ 
        P = Q1D
        Q = P1D
        stride0 = 1
        stride1 = P1D
    if version == int(not INTERLEAVE) | int(not TRANSPOSE):
        loopyCode += """
                     u_stride := ncomp*elemsize
                     v_stride := nqpt
                     u_comp_stride := elemsize
                     v_comp_stride := nelem * nqpt
                     u_size := elemsize
                     """
        u_stride = ncomp*elemsize
        v_stride = nqpt
        u_size = elemsize
    elif version == int(not INTERLEAVE) | TRANSPOSE:
        loopyCode += """
                     u_stride := nqpt
                     v_stride := ncomp*elemsize
                     u_comp_stride := nelem*nqpt
                     v_comp_stride := elemsize
                     u_size := nqpt
                     """
        u_stride = nqpt
        v_stride = ncomp*elemsize
        u_size = nqpt
    elif version == INTERLEAVE | int(not TRANSPOSE):
        loopyCode += """
                     u_stride := ncomp*elemsize
                     v_stride := ncomp*nqpt                     
                     """
        u_stride = ncomp*elemsize
        v_stride = ncomp*nqpt
    elif version == INTERLEAVE | TRANSPOSE:
        loopyCode += """
                     u_stride := ncomp*nqpt
                     v_stride := ncomp*elemsize                     
                     """
        u_stride = ncomp*nqpt
        v_stride = ncomp*elemsize
    else:
        print("ERROR")

    if not version & INTERLEAVE:
        loopyCode += """
                     u_offset := elem*u_stride + comp*u_comp_stride
                     v_offset := elem*v_stride + comp*v_comp_stride 
                     pre(d) := u_size*(P**(one*(dim-1-d)))
                     """
        pre = lambda d: u_size*P**(dim-1-d)
    else:
        loopyCode += """
                     u_offset := elem*u_stride
                     v_offset := elem*v_stride
                     pre(d) := u_stride*P**(dim-1-d)
                     """
        pre = lambda d: u_stride*P**(dim-1-d)
    loopyCode += """
                 post(d) := Q**d
                 c(d,k) := k % post(d)
                 j(d,k) := (k / post(d)) % Q
                 a(d,k) := k / (post(d) * Q)
                 <> PP = P
                 <float32> one = 1.0
                 """
    post = lambda d: Q**d
    iterVars =  ["{ [elem]: 0<=elem<nelem }",
                "{ [comp]: 0<=comp<ncomp }"]

    for d in range(constants["dim"]):
        writeLenStr = "writeLen{0}".format(str(d))
        constants[writeLenStr] = Q * post(d) * pre(d)

    if "dim" in constants and version==0:

        writeLenStrs = ""
        loopBodyStrs = ""
        for d in range(constants["dim"]):
            iterVars += ["{" + "[a{0}]: 0<=a{0}<{1}".format(str(d), str(pre(d))) + "}",
                         "{" + "[b{0}]: 0<=b{0}<{1}".format(str(d),str(P))+ "}",
                         "{" + "[c{0}]: 0<=c{0}<{1}".format(str(d),str(Q)) + "}",
                         "{" + "[d{0}]: 0<=d{0}<{1}".format(str(d), str(post(d))) + "}"]

            #loopBodyStrs += "for a{0},c{0},d{0}\n".format(str(d))

            tmpStr2 = "u" if d == 0 else "tmp{0}".format(str(d-1))
            rhs = "sum(b{0}, interp1d[c{0},b{0}]*{1}[a{0},b{0},c{0}])".format(str(d),tmpStr2)

            # Form lhs
            #br = "<> " if d <= 1 else ""
            tmpStr = "v[comp,elem," if d == constants["dim"] - 1 else "<> tmp{0}[".format(str(d))
            # Not needed if index into v? 
            #vo = ""#" + v_offset" if d == constants["dim"] - 1 else ""
            lhs = ("{1}a{0},d{0},c{0}]").format(str(d),tmpStr) 

            # Merge lhs and rhs
            loopBodyStrs += lhs + " = " + rhs + "\n"
            #loopBodyStrs += "end\n"  

        loopyCode += writeLenStrs + "for comp, elem\n" + loopBodyStrs + "end\n" 

    elif "dim" in constants and constants["dim"] <= 3:
        
        writeLenStrs = ""
        loopBodyStrs = ""
        for d in range(constants["dim"]):
            # Iteration variables
            #writeLenStrs += "<> writeLen{0} = Q * post({0}) * pre({0}))\n".format(str(d))
            iterVars += ["{" + "[k{0}]: 0<=k{0}<writeLen{0}".format(str(d)) + "}"]
            
            # Form rhs
            tmpStr2 = "u" if d == 0 else ("tmp2" if d%2 else "tmp")
            rhs = "sum(b, interp1d[j({0},k{0})*stride0 + b*stride1] * {1}[(a({0},k{0})*P + b)*post({0}) + c({0},k{0})])".format(str(d), tmpStr2)    

            # Form lhs
            tmpStr = "v" if d == constants["dim"] - 1 else ("<> tmp" if d%2 else "<> tmp2")
            vo = "v_offset + " if d == constants["dim"] - 1 else ""
            lhs = ("{1}[" + vo + "k{0}] = ").format(str(d),tmpStr)

            # Merge lhs and rhs
            loopBodyStrs += lhs + rhs + "\n" 
        
        loopyCode += writeLenStrs + "for comp, elem\n" + loopBodyStrs + "end\n"
     
    else:
        iterVars += ["{ [k]: 0<=k<writeLen }", "{ [kk]: 0<=kk<writeLen2}","{ [d]: 0<=d<dim-1 }"]
        loopyCode += """ 
                 <> writeLen2 = pre(dim-1) * post(dim-1) * Q
                 for elem, comp     
                    <> writeLen = pre(d) * post(d) * Q
                    if d == 0 
                        <> tmp2[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * u[u_offset + (a(d,k)*P + b)*post(d) + c(d,k)])
                    elif d%2 == 0
                        tmp2[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * tmp[(a(d,k)*P + b)*post(d) + c(d,k)])
                    else 
                        <> tmp[k] = sum(b, interp1d[j(d,k)*stride0 + b*stride1] * tmp2[(a(d,k)*P + b)*post(d) + c(d,k)])
                    end

                    if dim - 1 == 0 
                        v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                    elif dim-1 % 2 == 0
                        v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                    else
                        v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp2[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                    end
                 end
                 """
    print(loopyCode)
    kInterp = lp.make_kernel(
        iterVars,
        loopyCode,
        name="kInterp",
        target=target,
        kernel_data=kernel_data
    )

    kInterp = lp.fix_parameters(kInterp, **constants)
    #kInterp = lp.make_reduction_inames_unique(kInterp, "b")
    code = lp.generate_code_v2(kInterp).device_code()
    print(code)
    return(code)

def generate_kGrad(constants={},version=0,arch="INTEL_CPU",target=lp.OpenCLTarget(),fp_format=np.float64):

    kernel_data = [
        lp.GlobalArg("interp1d", fp_format),
        lp.GlobalArg("grad1d", fp_format),
        lp.GlobalArg("u", fp_format),
        lp.GlobalArg("v", fp_format),
        lp.GlobalArg("tmp", fp_format),
        lp.GlobalArg("tmp2", fp_format), 
        lp.ValueArg("nelem", np.int32)
    ]
    if constants=={}:
        kernel_data += [
            lp.ValueArg("elemsize", np.int32),
            lp.ValueArg("ncomp", np.int32),
            lp.ValueArg("nqpt", np.int32),
            lp.ValueArg("dim", np.int32),
            lp.ValueArg("P1D", np.int32),
            lp.ValueArg("Q1D", np.int32)
    ]

    

    # Remove this when finished testing
    if not "dim" in constants:
        constants["dim"] = 3

    loopyCode = ""

    if not version & TRANSPOSE: #Not transpose
        loopyCode += """
                     P := P1D
                     Q := Q1D
                     stride0 := P1D
                     stride1 := 1
                     """
    else: #Transpose
        loopyCode += """
                     P := Q1D
                     Q := P1D
                     stride0 := 1
                     stride1 := P1D
                     """ 

    if version == int(not INTERLEAVE) | int(not TRANSPOSE):
        loopyCode += """
                     u_stride := ncomp*elemsize
                     v_stride := nqpt
                     u_comp_stride := elemsize
                     v_comp_stride := nelem * nqpt
                     u_size := elemsize
                     u_dim_stride := 0
                     v_dim_stride := nelem*nqpt*ncomp
                     """
    elif version == int(not INTERLEAVE) | TRANSPOSE:
        loopyCode += """
                     u_stride := nqpt
                     v_stride := ncomp*elemsize
                     u_comp_stride := nelem*nqpt
                     v_comp_stride := elemsize
                     u_size := nqpt
                     u_dim_stride := nelem*nqpt*ncomp
                     v_dim_stride := 0
                     """
    elif version == INTERLEAVE | int(not TRANSPOSE):
        loopyCode += """
                     u_stride := ncomp*elemsize
                     pre_coef := ncomp*elemsize
                     v_stride := ncomp*nqpt*dim                     
                     u_offset := elem*u_stride
                     v_offset := elem*v_stride + d1*nqpt*ncomp
                     """
    elif version == INTERLEAVE | TRANSPOSE:
        loopyCode += """
                     u_stride := ncomp*nqpt*dim
                     pre_coef := ncomp*nqpt
                     v_stride := ncomp*elemsize                     
                     u_offset := elem*u_stride + d1*nqpt*ncomp
                     v_offset := elem*v_stride
                     """
    else:
        print("ERROR")

    if not version & INTERLEAVE:
        loopyCode += """
                     u_offset := elem*u_stride + d1*u_dim_stride + comp*u_comp_stride
                     v_offset := elem*v_stride + d1*v_dim_stride + comp*v_comp_stride 
                     pre_f(d) := u_size*((P)**(one*(dim-1-d)))
                     pre(d) := u_size*((P)**((dim-1-d)))
                     """
    else:
        loopyCode += """
                     pre_f(d) := pre_coef*((P)**(one*(dim-1-d)))
                     pre(d) := pre_coef*((P)**((dim-1-d)))
                     """

    loopyCode += """
                 post(d) := (Q)**(d)
                 post_f(d) := Q**(one*d)
                 c(d,k) := k % post(d)
                 j(d,k) := (k / post(d)) % Q
                 a(d,k) := k / (post(d) * Q)
                 <> PP = P
                 <float> one = 1.0f
                 """

    iterVars =  ["{ [elem]: 0<=elem<nelem }",
                "{ [comp]: 0<=comp<ncomp }",
                "{ [d1]: 0<=d1<dim }",
                "{ [b]: 0<=b<PP}"]
 

    if "dim" in constants and constants["dim"] <= 3:



        writeLenStrs = ""
        loopBodyStrs = ""
        dim = constants["dim"]
        for d1 in range(constants["dim"]):
            writeLenStrs += "<int> writeLen{0} = pre_f({0}) * post({0}) * Q\n".format(str(d1))
            for d2 in range(constants["dim"]):
                kStr = "k" + str(d1) + str(d2)
                tmpStr = "tmp" if d2%2 else "tmp2"
                tmpStr2 = "u" if d2 == 0 else ("tmp2" if d2%2 else "tmp")
                op = "grad1d" if d1 == d2 else "interp1d"

                # RHS
                rhs = "sum(b, {3}[j({1},{0})*stride0 + b*stride1] * {2}[(a({1},{0})*P + b)*post({1}) + c({1},{0})])".format(kStr, str(d2), tmpStr2, op) 
                if d2 == constants["dim"] - 1 and version & TRANSPOSE:
                    rhs = " v[v_offset + {0}] + ".format(kStr) + rhs

                # LHS
                if d2 == constants["dim"] - 1:
                    lhs = "v[v_offset + {0}] = ".format(kStr)
                else:
                    lhs = "{1}[{0}] = ".format(kStr,tmpStr)
          
                if d1==0 and d2 == 0:
                    tag = " {id=iter_" + str(dim*d1 + d2) + "}"
                else:
                    tag = " {id=iter_" + str(dim*d1 + d2) + ", dep=iter_" + str(dim*d1+d2-1) + "}"

                iterVarStr = "[{0}]: 0<={0}<{1}".format(kStr, "writeLen" + str(d2))
                iterVars += ["{" + iterVarStr + "}"]

                loopBodyStrs += lhs + rhs + tag + "\n"


        loopyCode += writeLenStrs
        loopyCode += """
                     for elem, comp, d1
                     """

        loopyCode += loopBodyStrs
        loopyCode += """
                     end
                     """
        
    else: #Not tested
        iterVars += ["{ [k]: 0<=k<writeLen }", "{ [kk]: 0<=kk<writeLen2}"]
        loopyCode += """ 
                     <> writeLen2 = pre(dim-1) * post(dim-1) * Q
                     for elem, comp     
                        <> writeLen = pre(d2) * post(d2) * Q
                        if d2 != d1
                            if d2 == 0 
                                <> tmp2[k] = sum(b, interp1d[j(d2,k)*stride0 + b*stride1] * u[u_offset + (a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            elif d2%2 == 0
                                tmp2[k] = sum(b, interp1d[j(d2,k)*stride0 + b*stride1] * tmp[(a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            else 
                                <> tmp[k] = sum(b, interp1d[j(d2,k)*stride0 + b*stride1] * tmp2[(a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            end
                        else
                            if d2 == 0 
                                tmp2[k] = sum(b, grad1d[j(d2,k)*stride0 + b*stride1] * u[u_offset + (a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            elif d2%2 == 0
                                tmp2[k] = sum(b, grad1d[j(d2,k)*stride0 + b*stride1] * tmp[(a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            else 
                                tmp[k] = sum(b, grad1d[j(d2,k)*stride0 + b*stride1] * tmp2[(a(d2,k)*P + b)*post(d2) + c(d2,k)])
                            end
                        end
                     """
        if not version & TRANSPOSE:
            loopyCode += """
                         if d2 != d1
                            if dim - 1 == 0 
                                v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            elif dim-1 % 2 == 0
                                v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            else
                                v[v_offset + kk] = sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp2[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            end
                         else
                            if dim - 1 == 0 
                                v[v_offset + kk] = sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            elif dim-1 % 2 == 0
                                v[v_offset + kk] = sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * tmp[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            else
                                v[v_offset + kk] = sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * tmp2[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            end
                         end 
                         end
                         """
        else:
            loopyCode += """
                        if d2 != d1
                            if dim - 1 == 0 
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            elif dim-1 % 2 == 0
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            else
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, interp1d[j(dim-1,kk)*stride0 + b*stride1] * tmp2[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            end
                         else
                            if dim - 1 == 0 
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * u[u_offset + (a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            elif dim-1 % 2 == 0
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * tmp[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            else
                                v[v_offset + kk] = v[v_offset + kk] + sum(b, grad1d[j(dim-1,kk)*stride0 + b*stride1] * tmp2[(a(dim-1,kk)*P + b)*post(dim-1) + c(dim-1,kk)])
                            end
                         end
                         end
                         """

    kGrad = lp.make_kernel(
        iterVars,
        loopyCode,
        name="kGrad",
        target=target,
        kernel_data=kernel_data
    )

    kGrad = lp.fix_parameters(kGrad, **constants)
    kGrad = lp.make_reduction_inames_unique(kGrad, "b")
    code = lp.generate_code_v2(kGrad).device_code()
    print(code)
    return code


def generate_kWeight(constants={},version=3,arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):

    dim = version
    kernel_data= [
        lp.GlobalArg("qweight1d", fp_format),
        lp.GlobalArg("w", fp_format),
        lp.ValueArg("nelem", np.int32)
    ]

    if constants=={}:
        kernel_data += [
            lp.ValueArg("Q1D", np.int32),
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

    code = lp.generate_code_v2(kWeight).device_code()
    print(code)
    return code

constants = {"elemsize": 3, "ncomp": 3, "nqpt": 3, "dim": 3, "P1D": 3, "Q1D": 3}
for version in range(0,1):
    generate_kInterp(constants=constants,version=version)
