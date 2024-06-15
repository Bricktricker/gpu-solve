add_executable(GpuSolve "main.cpp" "CpuGridData.cpp" "CpuSolver.cpp" "Vector3.cpp" "SyclGridData.cpp" "SyclSolver.cpp")
target_link_libraries(GpuSolve sycl-gtx)
include_directories(${CMAKE_SOURCE_DIR}/extern/sycl-gtx/sycl-gtx/include)
target_link_libraries(GpuSolve OpenCL::OpenCL)