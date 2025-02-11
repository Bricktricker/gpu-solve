#pragma once
#include "CpuGridData.h"

class NewtonSolver {
public:
	static void solve(CpuGridData& grid);

private:
	static void findError(CpuGridData& grid);
	static void compF(CpuGridData& grid);
};
