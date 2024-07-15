#pragma once
#include "SyclGridData.h"

class SyclSolver {
public:
	static void solve(SyclGridData& grid);

private:
	static double vcycle(cl::sycl::queue& queue, SyclGridData& grid);
	static void jacobi(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter);
	static void compResidual(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum);
	static double sumResidual(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum);
	static void restrict(cl::sycl::queue& queue, SyclBuffer& fine, SyclBuffer& coarse);
	static void interpolate(cl::sycl::queue& queue, SyclBuffer& fine, SyclBuffer& coarse);
};