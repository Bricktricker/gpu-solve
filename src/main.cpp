#include <iostream>
#include <filesystem>
#include <fstream>
#include "gridParams.h"
#include "CpuGridData.h"
#include "CpuSolver.h"

int main(int argc, char* argv[]) {
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
        for (std::size_t i = 0; i < gridParams.stencil.values.size(); i++) {
            configFile >> gridParams.stencil.values[i];
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
