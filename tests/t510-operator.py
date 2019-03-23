import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import pyopencl.clrandom
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

#----
lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

def generate_setup(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kernel_data = ["ctx", "Q", "iOf7", "oOf7", "in", "out"]
    dtypes={
        "in": fp_format,
        "ctx": np.int32,
        "oOf7": np.int32,
        "iOf7": np.int32
        }
 
    setup = lp.make_kernel(
        "{ [i]: 0<=i<Q }",
        """
        if false
            <> dummy = ctx[0] # Need to figure out how to remove
        end
        out[oOf7[0]+i] = in[iOf7[0]+i] * (in[iOf7[1]+i+Q*0]*in[iOf7[1]+i+Q*3] -
                                      in[iOf7[1]+i+Q*1]*in[iOf7[1]+i+Q*2])
        """,
        name="setup",
        assumptions="Q >= 0",
        kernel_data=kernel_data,
        target=target
        )

    setup = lp.fix_parameters(setup, **constants)
    setup = lp.add_and_infer_dtypes(setup, dtypes)

    return setup

def generate_mass(constants={}, arch="INTEL_CPU", fp_format=np.float64, target=lp.OpenCLTarget()):
    kernel_data = ["ctx", "Q", "iOf7", "oOf7", "in", "out"]
    dtypes={
        "in": fp_format,
        "ctx": np.int32,
        "oOf7": np.int32,
        "iOf7": np.int32
    }

    mass = lp.make_kernel(
    "{ [i]: 0<=i<Q }",
    """
    if false
        <> dummy = ctx[0]
    end
    out[i + oOf7[0]] = in[i + iOf7[0]] * in[i + iOf7[1]]
    """,
    name="mass",
    assumptions="Q >= 0",
    kernel_data=kernel_data,
    target=target
    )

    mass = lp.fix_parameters(mass, **constants)
    mass = lp.add_and_infer_dtypes(mass, dtypes)

    return mass

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
try:
    print(code)
except IOError:
    print('An IO error occured.')
except:
    print('An unknown error occured.')
