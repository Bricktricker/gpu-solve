#include <iostream>
#include <filesystem>
#include <fstream>
#include "gridParams.h"
#ifndef GPUSOLVE_CPU
    #include "sycl/ContextHandles.h"
    #include "sycl/SyclSolver.h"
    #include "sycl/NewtonSolver.h"
#else
    #include "cpu/CpuGridData.h"
    #include "cpu/CpuSolver.h"
    #include "cpu/NewtonSolver.h"
#endif

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Missing config file. Usage program.exe path/to/config.conf\n";
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

        int mode;
        configFile >> mode;
        gridParams.mode = static_cast<GridParams::Mode>(mode);

        configFile >> gridParams.preSmoothing;
        configFile >> gridParams.postSmoothing;
        configFile >> gridParams.omega;
        configFile >> gridParams.gamma;

        // read stencil
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

        gridParams.h = 1.0 / (gridParams.gridDim[1] + 1);
    }


#ifdef GPUSOLVE_CPU
    CpuGridData cpuGridData(gridParams);
    if (gridParams.mode == GridParams::Mode::NEWTON) {
        NewtonSolver::solve(cpuGridData);
    }else {
        CpuSolver::solve(cpuGridData);
    }
#else
    ContextHandles contextHandles = ContextHandles::init();
    SyclGridData syclGridData(gridParams);
    syclGridData.initBuffers(contextHandles.queue);

    if (gridParams.mode == GridParams::Mode::NEWTON) {
        NewtonSolver::solve(contextHandles.queue, syclGridData);
    }else {
        SyclSolver::solve(contextHandles.queue, syclGridData);
    }

#endif

    return 0;
}
