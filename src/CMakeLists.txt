add_executable(GpuSolve "main.cpp" "cpu/CpuGridData.cpp" "cpu/CpuSolver.cpp" "cpu/Vector3.cpp" "sycl/SyclGridData.cpp" "sycl/SyclSolver.cpp")
target_link_libraries(GpuSolve sycl-gtx)
include_directories(${CMAKE_SOURCE_DIR}/extern/sycl-gtx/sycl-gtx/include)
target_link_libraries(GpuSolve OpenCL::OpenCL)