#pragma once
#include "SyclGridData.h"

class NewtonSolver {
public:
	static void solve(cl::sycl::queue& queue, SyclGridData& grid);

private:
	static double compF(cl::sycl::queue& queue, SyclGridData& grid, bool calcSum);
	static void findError(cl::sycl::queue& queue, SyclGridData& grid);
};
