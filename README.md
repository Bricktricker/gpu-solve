# GPU Solve

Code from my masters thesis. Solves a 3D system of linear equations on the CPU and GPU using SYCL.

The sycl solver mainly uses the [ProGTX/sycl-gtx](https://github.com/ProGTX/sycl-gtx) library to compile the SYCL kernels to OpenCL kernels during execution. The library is included in this project in the [extern/sycl-gtx](https://github.com/Bricktricker/gpu-solve/tree/main/extern/sycl-gtx) folder. I have modified some parts of the code to improve performance.

## Build
```
git clone https://github.com/Bricktricker/gpu-solve.git
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make GpuSolve-cpu GpuSolve-gtx
```
If you want to use a diffrent SYCL implementation, build the `GpuSolve-sycl` target and adjust the compiler settings beforehand.

## Usage
After building, the executables can be found in the `build/src/` directory. Launch the application with `./GpuSolve-cpu <path/to/config>`. 

The config file describes the problem structure, an example can be found at [examples/data-2nd_order.conf](https://github.com/Bricktricker/gpu-solve/blob/main/examples/data-2nd_order.conf). Description of the config file lines:
1. Number of v-cycles
2. Tolerance, currently unused
3. X dimension of the grid
4. Y dimension of the grid
5. Z dimension of the grid
6. Switch between linear (1) and non-linear (0) problem
7. Number of pre-smoothing steps
8. Number of post-smoothing steps
9. Relaxation coefficient
10. The seven stencil values, seperated by space
11. Stencil value offsets in the X direction
12. Stencil value offsets in the Y direction
13. Stencil value offsets in the Z direction
