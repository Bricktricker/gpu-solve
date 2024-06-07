#pragma once
#include "CpuGridData.h"

class CpuSolver {
public:

	static void solve(CpuGridData& grid);

private:
	static double compResidual(const CpuGridData& grid, std::size_t level);
	static std::vector<double> compResidualVec(const CpuGridData& grid, std::size_t level);

	static double vcycle(const CpuGridData& grid);
	static double jacobi(CpuGridData& grid, std::size_t level, std::size_t maxiter);
};
