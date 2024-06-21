#pragma once
#include "SyclGridData.h"

class SyclSolver {
public:
	static void solve(SyclGridData& grid);

private:
	static void vsycle(cl::sycl::handler& cgh, SyclGridData& grid);
	static void jacobi(cl::sycl::handler& cgh, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter);
	static void compResidual(cl::sycl::handler& cgh, SyclGridData& grid, std::size_t levelNum);
	static float sumResidual(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum);
};