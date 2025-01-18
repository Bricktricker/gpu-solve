#include "NewtonSolver.h"
#include "CpuSolver.h"
#include "../Timer.h"
#include <iostream>

void NewtonSolver::solve(CpuGridData& grid) {
	// Compute inital residual

	grid.mode = GridParams::Mode::NONLINEAR;
	double initialResidual = CpuSolver::compResidual(grid, 0);
	std::cout << "Inital residual: " << initialResidual << '\n';

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();

		grid.mode = GridParams::Mode::NONLINEAR;
		double res = CpuSolver::compResidual(grid, 0);
		std::cout << "residual before findError: " << res << '\n';
		grid.mode = GridParams::Mode::NEWTON;

		findError(grid);

		//std::cout << "iter: " << i << " residual: " << res << ' ';
		Timer::stop();
	}

}

void NewtonSolver::findError(CpuGridData& grid)
{
	CpuGridData mgGrid = grid; // make a copy for MultiGrid;
	CpuGridData::LevelData& rootLevel = mgGrid.getLevel(0);

	// Solve f = J(v)*e, where f is the residual r
	rootLevel.f = rootLevel.r;
	rootLevel.v.fill(0.0);

	CpuSolver::solve(mgGrid);

	Vector3& v = grid.getLevel(0).v;
	v += mgGrid.getLevel(0).v;
}
