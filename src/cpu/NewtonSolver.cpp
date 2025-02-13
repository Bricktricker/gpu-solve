#include "NewtonSolver.h"
#include "CpuSolver.h"
#include "../Timer.h"
#include <iostream>
#ifdef _WIN32
	#include <windows.h>
	#include <psapi.h>
#endif

void NewtonSolver::solve(CpuGridData& grid) {
	// Compute inital residual

	grid.mode = GridParams::Mode::NONLINEAR;
	double initialResidual = CpuSolver::compResidual(grid, 0);
	std::cout << "Inital residual: " << initialResidual << '\n';
	grid.mode = GridParams::Mode::NEWTON;

	// Stores the original right hand side, never gets changed
	grid.newtonF = grid.getLevel(0).f;

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();
		
		compF(grid);
		grid.getLevel(0).v.fill(0.0);

		findError(grid);

		// TODO: figure out how to compute the correct residual
		// fnorm, or this?
		grid.mode = GridParams::Mode::NONLINEAR;
		grid.getLevel(0).v.fill(0.0);
		compF(grid);

		double res = CpuSolver::compResidual(grid, 0);
		grid.mode = GridParams::Mode::NEWTON;

		std::cout << "iter: " << i << " residual: " << res << ' ';
		Timer::stop();

#ifdef _WIN32
		::PROCESS_MEMORY_COUNTERS pmc = {};
		if (::GetProcessMemoryInfo(::GetCurrentProcess(), &pmc, sizeof(pmc))) {
			std::cout << "Current ram usage: " << pmc.WorkingSetSize << '\n';
		}
#endif

	}

	// Result is stored in level_0.newtonV
}

// computes the residual using newtonV and the original right hand side (newtonF)
// stores the result in level_0.f
void NewtonSolver::compF(CpuGridData& grid)
{
	CpuGridData::LevelData& level = grid.getLevel(0);

	double Fnorm = 0.0;

#pragma omp parallel for schedule(static,8) reduction(+:Fnorm)
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

				Fnorm += f * f;
			}
		}
	}

	std::cout << "Fnorm: " << Fnorm << std::endl;
}

void NewtonSolver::findError(CpuGridData& grid)
{
	// Solve f = J(v)*e, where f is the residual r, computed from the current newtonV and the original right hand side

	// restrict newtonV to all levels
	for (std::size_t i = 1; i < grid.numLevels() - 1; i++) {
		const Vector3& src = grid.getLevel(i - 1).newtonV;
		Vector3& dst = grid.getLevel(i).newtonV;
		CpuSolver::restrict(src, dst);
	}

	grid.printProgress = false;
	CpuSolver::solve(grid);
	grid.printProgress = true;

	Vector3& newtonV = grid.getLevel(0).newtonV;
	newtonV += grid.getLevel(0).v;
}
