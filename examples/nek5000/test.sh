#!/bin/sh
./make-nek-examples.sh
wait $!
./run-nek-examples.sh -b 0 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 1 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 2 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 3 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 4 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 5 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 6 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 7 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 8 -e bp1 -c /gpu/opencl/nvidia
wait $!
./run-nek-examples.sh -b 9 -e bp1 -c /gpu/opencl/nvidia
wait $!
