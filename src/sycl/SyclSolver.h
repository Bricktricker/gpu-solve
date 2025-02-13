#pragma once
#include "SyclGridData.h"

class SyclSolver {
public:
	static void solve(cl::sycl::queue& queue, SyclGridData& grid);
	static double sumResidual(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum);
	static void restrict(cl::sycl::queue& queue, SyclBuffer& fine, SyclBuffer& coarse);

private:
	static double vcycle(cl::sycl::queue& queue, SyclGridData& grid);
	static void jacobi(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter);
	static void compResidual(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum);
	static void applyStencil(cl::sycl::queue& queue, SyclGridData& grid, std::size_t level, SyclBuffer& v);
	static void interpolate(cl::sycl::queue& queue, SyclBuffer& fine, SyclBuffer& coarse);
};