set(BASE_CPP_FILES "main.cpp" "cpu/Vector3.cpp" "Timer.cpp")
set(BASE_SYCL_FILES ${BASE_CPP_FILES} "sycl/SyclGridData.cpp" "sycl/SyclSolver.cpp" "sycl/NewtonSolver.cpp")

add_executable(GpuSolve-cpu ${BASE_CPP_FILES} "cpu/CpuGridData.cpp" "cpu/CpuSolver.cpp" "cpu/NewtonSolver.cpp")
target_compile_definitions(GpuSolve-cpu PUBLIC GPUSOLVE_CPU)
if(OpenMP_CXX_FOUND)
    target_link_libraries(GpuSolve-cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(GpuSolve-gtx ${BASE_SYCL_FILES})
target_link_libraries(GpuSolve-gtx sycl-gtx)
target_include_directories(GpuSolve-gtx PRIVATE ${CMAKE_SOURCE_DIR}/extern/sycl-gtx/sycl-gtx/include)
target_link_libraries(GpuSolve-gtx OpenCL::OpenCL)

add_executable(GpuSolve-sycl ${BASE_SYCL_FILES})
target_link_libraries(GpuSolve-sycl PRIVATE sycl)