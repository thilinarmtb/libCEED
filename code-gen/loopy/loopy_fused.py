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

def generate_fused_kernel():
    massf = lp.make_kernel("{ [iter]: 0<=iter<top }",
    """
        
    """,
    name="fused",
    target=lp.OpenCLTarget()



#def generate_fused_kernel(rk, bk, qk):
    

