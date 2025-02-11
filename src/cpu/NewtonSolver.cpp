#include "NewtonSolver.h"
#include "CpuSolver.h"
#include "../Timer.h"
#include <iostream>

void NewtonSolver::solve(CpuGridData& grid) {
	// Compute inital residual

	grid.mode = GridParams::Mode::NONLINEAR;
	double initialResidual = CpuSolver::compResidual(grid, 0);
	std::cout << "Inital residual in NewtonSolver: " << initialResidual << '\n';

	grid.newtonF = grid.getLevel(0).f;

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();
		
		compF(grid);

		grid.mode = GridParams::Mode::NONLINEAR;
		grid.getLevel(0).v.fill(0.0);
		double res = CpuSolver::compResidual(grid, 0);
		std::cout << "residual before findError: " << res << '\n';
		grid.mode = GridParams::Mode::NEWTON;

		findError(grid);

		//std::cout << "iter: " << i << " residual: " << res << ' ';
		Timer::stop();
	}

}

void NewtonSolver::compF(CpuGridData& grid)
{
	CpuGridData::LevelData& level = grid.getLevel(0);

//#pragma omp parallel for schedule(static,8) reduction(+:res)
	for (std::int64_t x = 1; x < level.levelDim[0]+1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1]+1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2]+1; z++) {

				double stencilsum = 0.0;
				for (std::size_t i = 0; i < grid.stencil.values.size(); i++) {
					double vVal = level.newtonV.get(x + grid.stencil.getXOffset(i), y + grid.stencil.getYOffset(i), z + grid.stencil.getZOffset(i));
					stencilsum += grid.stencil.values[i] * vVal;
				}

				stencilsum /= level.h * level.h;

				// See tutorial_multigrid.pdf, page 102, Formula 6.13
				double ex = exp(level.newtonV.get(x, y, z));
				double nonLinear = grid.gamma * level.newtonV.get(x, y, z) * ex;
				stencilsum += nonLinear;

				double f = grid.newtonF.get(x, y, z) - stencilsum;
				level.f.set(x, y, z, f);
			}
		}
	}
}

void NewtonSolver::findError(CpuGridData& grid)
{
	CpuGridData mgGrid = grid; // make a copy for MultiGrid;
	CpuGridData::LevelData& rootLevel = mgGrid.getLevel(0);

	// Solve f = J(v)*e, where f is the residual r

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
