set(BASE_CPP_FILES "main.cpp" "cpu/Vector3.cpp" "Stencil.cpp" "Timer.cpp")
set(BASE_SYCL_FILES ${BASE_CPP_FILES} "sycl/SyclGridData.cpp" "sycl/SyclSolver.cpp")

add_executable(GpuSolve-cpu ${BASE_CPP_FILES} "cpu/CpuGridData.cpp" "cpu/CpuSolver.cpp")
target_compile_definitions(GpuSolve-cpu PUBLIC GPUSOLVE_CPU)

add_executable(GpuSolve-gtx ${BASE_SYCL_FILES})
target_link_libraries(GpuSolve-gtx sycl-gtx)
target_include_directories(GpuSolve-gtx PRIVATE ${CMAKE_SOURCE_DIR}/extern/sycl-gtx/sycl-gtx/include)
target_link_libraries(GpuSolve-gtx OpenCL::OpenCL)
target_compile_definitions(GpuSolve-gtx PUBLIC SYCL_GTX_TARGET)

add_executable(GpuSolve-sycl ${BASE_SYCL_FILES})
target_link_libraries(GpuSolve-sycl PRIVATE sycl)