import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2

import sys
import json

lp.set_caching_enabled(False)
from warnings import filterwarnings, catch_warnings
filterwarnings('error', category=lp.LoopyWarning)
import loopy.options
loopy.options.ALLOW_TERMINAL_COLORS = False

from loopy_restrict import generate_kRestrict0, generate_kRestrict1, \
        generate_kRestrict2, generate_kRestrict6
from loopy_basis import generate_kZero, generate_kInterp, generate_kWeight, \
        generate_kGrad

def write_kernel(data):
    kernel = data["kernel"]
    workDim = data["work_dim"]
    globalWorkSize = data["global_work_size"]
    localWorkSize = data["local_work_size"]

    kernelLength = kernel.count('\n')

    print("[kernel_length]\n{}\n".format(kernelLenght))
    print("[kernel]\n{}\n".format(kernel))
    print("[work_dim]\n{}\n".format(workDim))
    print("[global_work_size]\n{}\n".format(globalWorkSize))
    print("[local_work_size]\n{}\n".format(localWorkSize))

arg_len = len(sys.argv)
if arg_len != 4:
    print("Usage: python loopy_kernel_output.py kernel_name arch '{\"c1\": val1, ... }'")
    print("Example: python loopy_kernel_output.py kRestrict0 '{\"elemsize\": 8, ... }'")
    sys.exit(1)

kernel_name = sys.argv[1]
arch = sys.argv[2]
constants = json.loads(sys.argv[3])

if kernel_name == 'kRestrict0':
    k = generate_kRestrict0(constants, arch)
elif kernel_name == 'kRestrict1':
    k = generate_kRestrict1(constants, arch)
elif kernel_name == 'kRestrict2':
    k = generate_kRestrict2(constants, arch)
elif kernel_name == 'kRestrict6':
    k = generate_kRestrict6(constants, arch)
elif kernel_name == 'kZero':
    k = generate_kZero(constants, arch)
elif kernel_name == 'kInterp':
    k = generate_kInterp(constants, arch)
elif kernel_name == 'kGrad':
    k = generate_kGrad(constants, arch)
elif kernel_name == 'kWeight':
    k = generate_kWeight(constants, arch)
else:
    print("Invalid kernel name: {}".format(kernel_name))
    sys.exit(1)

code = lp.generate_code_v2(k).device_code()
print(code)
