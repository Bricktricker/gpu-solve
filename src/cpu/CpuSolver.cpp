#include "CpuSolver.h"
#include <assert.h>
#include <iostream>
#include <chrono>
#include <math.h>

void CpuSolver::solve(CpuGridData& grid)
{
	// Compute inital residual
	// TODO: if(grid.periodic) { updateResidual(f & v) }, needed?
	double initialResidual = compResidual(grid, 0);
	std::cout << "Inital residual: " << initialResidual << '\n';

	auto start = std::chrono::high_resolution_clock::now();

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		double res = vcycle(grid);

		const auto end = std::chrono::high_resolution_clock::now();
		const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		start = end;

		std::cout << "iter: " << i << " residual: " << res << " took " << time << "ms\n";
	}
}

double CpuSolver::compResidual(CpuGridData& grid, std::size_t levelNum)
{
	double res = 0.0;
	CpuGridData::LevelData& level = grid.getLevel(levelNum);

	for (std::size_t x = 1; x < level.levelDim[0]+1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1]+1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2]+1; z++) {
				
				double stencilsum = 0.0;
				for (std::size_t i = 0; i < level.stencil.values.size(); i++) {
					double vVal = level.v.get(x + level.stencil.getXOffset(i), y + level.stencil.getYOffset(i), z + level.stencil.getZOffset(i));
					stencilsum += level.stencil.values[i] * vVal;
				}

				double r = level.f.get(x, y, z) - stencilsum;
				level.r.set(x, y, z, r);
				res += r * r;
			}
		}
	}
	
	return sqrt(res);
}

double CpuSolver::vcycle(CpuGridData& grid)
{
	for (std::size_t i = 0; i < grid.numLevels()-1; i++) {
		jacobi(grid, i, grid.preSmoothing);

		// clear v for next level
		CpuGridData::LevelData& nextLevel = grid.getLevel(i + 1);
		nextLevel.v.fill(0.0);

		// compute residual
		compResidual(grid, i);
		Vector3& r = grid.getLevel(i).r;
		if (grid.periodic) updateGhosts(r);

		// restrict residual to next level f
		restrict(r, nextLevel.f);
		if (grid.periodic) updateGhosts(nextLevel.f);
	}
	
	// reached coarsed level, solve now
	jacobi(grid, grid.numLevels() - 1, grid.preSmoothing+grid.postSmoothing);

	for (std::size_t i = grid.numLevels() - 1; i > 0; i--) {
		// interpolate v to previos level e
		interpolate(grid, i - 1);

		// v = v + e
		auto& level = grid.getLevel(i - 1);
		level.v += level.e;

		jacobi(grid, i - 1, grid.postSmoothing);
	}

	// returns current residual
	return compResidual(grid, 0);
}

void CpuSolver::jacobi(CpuGridData& grid, std::size_t levelNum, std::size_t maxiter)
{	
	CpuGridData::LevelData& level = grid.getLevel(levelNum);
	const double alpha = 1.0 / level.stencil.values[0]; // stencil center

	for (std::size_t i = 0; i < maxiter; i++) {
		if(grid.periodic) updateGhosts(level.v);
		
		compResidual(grid, levelNum);
		
		for (std::size_t x = 1; x < level.levelDim[0] + 1; x++) {
			for (std::size_t y = 1; y < level.levelDim[1] + 1; y++) {
				for (std::size_t z = 1; z < level.levelDim[2] + 1; z++) {
					double newV = level.v.get(x, y, z) + grid.omega * (alpha * level.r.get(x, y, z));
					level.v.set(x, y, z, newV);
				}
			}
		}
	}

	if (grid.periodic) updateGhosts(level.v);
}

void CpuSolver::restrict(const Vector3& fine, Vector3& coarse)
{
	for (std::size_t x = 1; x < coarse.getXdim()-1; x++) {
		for (std::size_t y = 1; y < coarse.getYdim()-1; y++) {
			for (std::size_t z = 1; z < coarse.getZdim()-1; z++) {

				std::size_t xCenter = 2 * x;
				std::size_t yCenter = 2 * y;
				std::size_t zCenter = 2 * z;

				double coarseValue = 0.0;

				for (int ii = -2 + 1; ii < 2; ii++) {
					for (int jj = -2 + 1; jj < 2; jj++) {
						for (int kk = -2 + 1; kk < 2; kk++) {
							double fac = ((2.0 - abs(ii)) / 2.0) * ((2.0 - abs(jj)) / 2.0) * ((2.0 - abs(kk)) / 2.0);
							coarseValue += fac * fine.get(xCenter + ii, yCenter + jj, zCenter + kk);
						}
					}
				}

				coarse.set(x, y, z, coarseValue);
			}
		}
	}
}

void CpuSolver::interpolate(CpuGridData& grid, std::size_t level)
{
	const Vector3& coarse = grid.getLevel(level + 1).v;
	Vector3& fine = grid.getLevel(level).e;

	// prepare
	for (std::size_t x = 0; x < fine.getXdim() - 1; x += 2) {
		for (std::size_t y = 0; y < fine.getYdim() - 1; y += 2) {
			for (std::size_t z = 0; z < fine.getZdim() - 1; z += 2) {
				double val = coarse.get(x/2, y/2, z/2);
				fine.set(x, y, z, val);
			}
		}
	}

	if (grid.periodic) {
		updateGhosts(fine);
	}

	// Interpolate in x-direction
	for (std::size_t x = 0; x+2 < fine.getXdim(); x += 2) {
		for (std::size_t y = 0; y < fine.getYdim(); y += 2) {
			for (std::size_t z = 0; z < fine.getZdim(); z += 2) {
				double val = 0.5 * fine.get(x, y, z) + 0.5 * fine.get(x + 2, y, z);
				fine.set(x+1, y, z, val);
			}
		}
	}

	// Interpolate in y-direction
	for (std::size_t x = 0; x < fine.getXdim(); x++) {
		for (std::size_t y = 0; y + 2 < fine.getYdim(); y += 2) {
			for (std::size_t z = 0; z < fine.getZdim(); z += 2) {
				double val = 0.5 * fine.get(x, y, z) + 0.5 * fine.get(x, y+2, z);
				fine.set(x, y+1, z, val);
			}
		}
	}

	// Interpolate in z-direction
	for (std::size_t x = 0; x < fine.getXdim(); x++) {
		for (std::size_t y = 0; y < fine.getYdim(); y++) {
			for (std::size_t z = 0; z + 2 < fine.getZdim(); z += 2) {
				double val = 0.5 * fine.get(x, y, z) + 0.5 * fine.get(x, y, z + 2);
				fine.set(x, y, z+1, val);
			}
		}
	}

	if (grid.periodic) {
		updateGhosts(fine);
	}
}

void CpuSolver::updateGhosts(Vector3& vec)
{
	// must only be called when grid is periodic

	// z-dimension
	for (std::size_t x = 0; x < vec.getXdim(); x++) {
		for (std::size_t y = 0; y < vec.getYdim(); y++) {
			vec.set(x, y, 0, vec.get(x, y, vec.getZdim()-2));
			vec.set(x, y, vec.getZdim()-1, vec.get(x, y, 1));
		}
	}

	// y-dimension
	for (std::size_t x = 0; x < vec.getXdim(); x++) {
		for (std::size_t z = 0; z < vec.getZdim(); z++) {
			vec.set(x, 0, z, vec.get(x, vec.getYdim()-2, z));
			vec.set(x, vec.getYdim() - 1, z, vec.get(x, 1, z));
		}
	}

	// x-dimension
	for (std::size_t y = 0; y < vec.getYdim(); y++) {
		for (std::size_t z = 0; z < vec.getZdim(); z++) {
			vec.set(0, y, z, vec.get(vec.getXdim()-2, y, z));
			vec.set(vec.getXdim() - 1, y, z, vec.get(1, y, z));
		}
	}

}
