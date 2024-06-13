#include <iostream>
#include <filesystem>
#include <fstream>
#include "gridParams.h"
#include "CpuGridData.h"
#include "CpuSolver.h"

#define SYCL_SIMPLE_SWIZZLES
#include <cl/sycl.hpp>
#include <SYCL/detail/debug.h>
using namespace cl::sycl;

int main(int argc, char* argv[]) {

    auto platforms = cl::sycl::platform::get_platforms();
    std::cout << "Number of platforms: " << platforms.size() << '\n';
    cl::sycl::platform P = platforms.at(0);

    const auto devices = P.get_devices(cl::sycl::info::device_type::gpu);
    cl::sycl::device D = devices[0];

    std::cout << "Device name: " << D.get_info<cl::sycl::info::device::name>() << '\n';
    std::cout << "Vendor: " << D.get_info<cl::sycl::info::device::vendor>() << '\n';
    std::cout << "OpenCl version: " << D.get_info<cl::sycl::info::device::opencl_version>() << '\n';

    if (argc < 2) {
        std::cerr << "Missing config file. Usage program.exe path/to/config.conf";
        return 1;
    }

    const std::filesystem::path configFilePath = std::filesystem::path(argv[1]);
    if (!std::filesystem::exists(configFilePath) || !std::filesystem::is_regular_file(configFilePath)) {
        std::cerr << configFilePath << " does not exist or is not a file\n";
        return 1;
    }

    std::cout << "Using config file " << configFilePath << '\n';

    GridParams gridParams;

    {
        std::ifstream configFile(configFilePath);
        configFile >> gridParams.maxiter;
        configFile >> gridParams.tol;
        configFile >> gridParams.gridDim[0];
        configFile >> gridParams.gridDim[1];
        configFile >> gridParams.gridDim[2];
        configFile >> gridParams.periodic;
        configFile >> gridParams.preSmoothing;
        configFile >> gridParams.postSmoothing;
        configFile >> gridParams.omega;

        // read stencil        
        gridParams.stencil.values.resize(7);
        gridParams.stencil.offsets.resize(7);

        for (std::size_t i = 0; i < gridParams.stencil.values.size(); i++) {
            configFile >> gridParams.stencil.values[i];
        }

        for (std::size_t i = 0; i < gridParams.stencil.offsets.size(); i++) {
            int val;
            configFile >> val;
            std::get<0>(gridParams.stencil.offsets[i]) = val;
        }
        for (std::size_t i = 0; i < gridParams.stencil.offsets.size(); i++) {
            int val;
            configFile >> val;
            std::get<1>(gridParams.stencil.offsets[i]) = val;
        }
        for (std::size_t i = 0; i < gridParams.stencil.offsets.size(); i++) {
            int val;
            configFile >> val;
            std::get<2>(gridParams.stencil.offsets[i]) = val;
        }

        if (gridParams.periodic) {
            gridParams.h = 1.0 / gridParams.gridDim[1];
        }else {
            gridParams.h = 1.0 / (gridParams.gridDim[1]+1);
        }

    }

    // TODO: check for cpu or gpu implementation

    CpuGridData cpuGridData(gridParams);

    CpuSolver::solve(cpuGridData);

    return 0;
}
