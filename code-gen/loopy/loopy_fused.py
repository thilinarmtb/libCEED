import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

import loopy_basis as basis
import loopy_restrict as restrict


r_base = restrict.generate_kRestrict6()

kZero = basis.generate_kZero()
r = lp.rename_argument(r_base, "vv", "v")
print(r)
restrict_kZero_fused = lp.fuse_kernels([r,kZero], data_flow=[])
print(restrict_kZero_fused)

'''
kInterp = basis.generate_kInterp()
r = lp.rename_argument(r_base, "vv", "d_v")
restrict_kInterp_fused = lp.fuse_kernels([r, kInterp])
print(restrict_kInterp_fused)
'''


'''
def generate_fused_kernel():
    massf = lp.make_kernel("{ [iter]: 0<=iter<top }",
    """
        
    """,
    name="fused",
    target=lp.OpenCLTarget()



#def generate_fused_kernel(rk, bk, qk):
'''    

