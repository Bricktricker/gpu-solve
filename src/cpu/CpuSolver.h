#pragma once
#include "CpuGridData.h"

class CpuSolver {
public:

	static void solve(CpuGridData& grid);

private:
	static double compResidual(CpuGridData& grid, std::size_t level);
	static double vcycle(CpuGridData& grid);
	static void jacobi(CpuGridData& grid, std::size_t level, std::size_t maxiter);
	static void applyStencil(CpuGridData& grid, std::size_t level, const Vector3& v);
	static void restrict(const Vector3& src, Vector3& dst);
	static void interpolate(CpuGridData& grid, std::size_t level);
};
