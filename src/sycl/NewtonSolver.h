#pragma once
#include "SyclGridData.h"
#include "ContextHandles.h"

class NewtonSolver {
public:
	static void solve(ContextHandles& handles, SyclGridData& grid);

private:
	static void compF(cl::sycl::queue& queue, SyclGridData& grid);
	static void findError(ContextHandles& handles, SyclGridData& grid);
	static SyclGridData makeCopyForSolver(cl::sycl::queue& queue, SyclGridData& grid);
};
