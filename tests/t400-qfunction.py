import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

# setup
#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def generate_setup(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    setup = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        if false
            <> dummy = ctx[0] # Need to figure out how to remove
        end
        out[i + oOf7[0]] = in[i + iOf7[0]]
        """,
        name="t400_qfunction_setup",
        assumptions="Q >= 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=lp.OpenCLTarget()
    )

    return setup

def generate_mass(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    mass = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        if false
            <> dummy = ctx[0]
        end
        out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]]
        """,
        name="t400_qfunction_mass",
        assumptions="Q >= 0",
        kernel_data=["ctx", "Q", "iOf7", "oOf7", "in", "out"],
        target=lp.OpenCLTarget()
    )

kernel_name = sys.argv[1]
arch = sys.argv[2]
constants = json.loads(sys.argv[3])

if kernel_name == 'mass':
    k = generate_mass(constants, arch)
elif kernel_name == 'setup':
    k = generate_setup(constants, arch)
else:
    print("Invalid kernel name: {}".format(kernel_name))
    sys.exit(1)

code = lp.generate_code_v2(k).device_code()
print(code)
print()
