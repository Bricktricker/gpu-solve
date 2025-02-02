#include "NewtonSolver.h"
#include "CpuSolver.h"
#include "../Timer.h"
#include <iostream>

void NewtonSolver::solve(CpuGridData& grid) {
	// Compute inital residual
	//grid.gamma = 0.0;

	grid.mode = GridParams::Mode::NONLINEAR;
	double initialResidual = CpuSolver::compResidual(grid, 0);
	std::cout << "Inital residual in NewtonSolver: " << initialResidual << '\n';

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();

		grid.mode = GridParams::Mode::NONLINEAR;
		grid.getLevel(0).v = grid.getLevel(0).newtonV;
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

	// restrict newtonV to all levels
	for (std::size_t i = 1; i < grid.numLevels() - 1; i++) {
		const Vector3& src = mgGrid.getLevel(i - 1).newtonV;
		Vector3& dst = mgGrid.getLevel(i).newtonV;
		CpuSolver::restrict(src, dst);
	}

	CpuSolver::solve(mgGrid);

	Vector3& newtonV = grid.getLevel(0).newtonV;
	newtonV += mgGrid.getLevel(0).v;
}
