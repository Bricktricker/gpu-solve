#include "CpuSolver.h"
#include <assert.h>
#include <iostream>

void CpuSolver::solve(CpuGridData& grid)
{
	// Compute inital residual
	// TODO: if(grid.periodic) { updateResidual(f & v) }, needed?
	double initialResidual = compResidual(grid, 0);

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		double res = vcycle(grid);
		std::cout << "iter: " << i << " residual: " << res << '\n';

		break; // break for now, until the first itteration works
	}

	int frzu = 0; // debug break point
}

double CpuSolver::compResidual(const CpuGridData& grid, std::size_t levelNum)
{
	double res = 0.0;
	const CpuGridData::LevelData& level = grid.getLevel(levelNum);

	for (std::size_t x = 1; x < level.levelDim[0]+1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1]+1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2]+1; z++) {
				
				double stencilsum = 0.0;
				for (std::size_t i = 0; i < level.stencil.values.size(); i++) {
					double vVal = level.v.get(x + level.stencil.getXOffset(i), y + level.stencil.getYOffset(i), z + level.stencil.getZOffset(i));
					stencilsum += level.stencil.values[i] * vVal;
				}

				double r = level.f.get(x, y, z) - stencilsum;
				res += r * r;
			}
		}
	}
	
	return sqrt(res);
}

// like compResidual, but returns the vector r
Vector3 CpuSolver::compResidualVec(const CpuGridData& grid, std::size_t levelNum)
{
	const CpuGridData::LevelData& level = grid.getLevel(levelNum);

	Vector3 r(level.f.getXdim(), level.f.getYdim(), level.f.getZdim());

	for (std::size_t x = 1; x < level.levelDim[0] + 1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1] + 1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2] + 1; z++) {

				double stencilsum = 0.0;
				for (std::size_t i = 0; i < level.stencil.values.size(); i++) {
					double vVal = level.v.get(x + level.stencil.getXOffset(i), y + level.stencil.getYOffset(i), z + level.stencil.getZOffset(i));
					stencilsum += level.stencil.values[i] * vVal;
				}

				double rVal = level.f.get(x, y, z) - stencilsum;
				r.set(x, y, z, rVal);
			}
		}
	}
	
	return r;
}

double CpuSolver::vcycle(CpuGridData& grid)
{
	for (std::size_t i = 0; i < grid.numLevels()-1; i++) {
		double res = jacobi(grid, i, grid.preSmoothing);
		std::cout << "jacobi(" << i << ")= " << res << "\n";

		// clear v for next level
		CpuGridData::LevelData& nextLevel = grid.getLevel(i + 1);
		nextLevel.v.fill(0.0);
		std::cout << "v" << (i + 1) << " = 0 // clear old v\n";

		// compute residual
		Vector3 r = compResidualVec(grid, i);
		if (grid.periodic) updateGhosts(r);

		// validated r for level 0

		// restrict residual to next level f
		std::cout << "f" << (i + 1) << " = restrict(r" << i << ")\n";
		restrict(r, nextLevel.f);
		if (grid.periodic) updateGhosts(nextLevel.f);

		// valdiated nextLevel.f for level 0
	}
	
	// reached coarsed level, solve now
	double topJac = jacobi(grid, grid.numLevels() - 1, grid.preSmoothing+grid.postSmoothing);
	std::cout << "top jacobi(" << grid.numLevels() - 1 << ") = " << topJac << '\n';

	for (std::size_t i = grid.numLevels() - 1; i > 0; i--) {
		// interpolate v to previos level e
		std::cout << "e" << (i - 1) << " = interpolate(v" << i << ")\n";
		Vector3 e = interpolate(grid.getLevel(i).v);

		// v = v + e

		std::cout << 'v' << i - 1 << " += e" << i - 1 << '\n';
		Vector3& v = grid.getLevel(i - 1).v;
		v += e;

		double bottomJac = jacobi(grid, i - 1, grid.postSmoothing);
		std::cout << "jacobi(" << i - 1 << ") = " << bottomJac << '\n';
	}

	// returns current residual
	return compResidual(grid, 0);
}

double CpuSolver::jacobi(CpuGridData& grid, std::size_t levelNum, std::size_t maxiter)
{
	// Validated jacobi for level 0
	
	CpuGridData::LevelData& level = grid.getLevel(levelNum);
	const double alpha = 1.0 / level.stencil.values[0]; // stencil center

	for (std::size_t i = 0; i < maxiter; i++) {
		if(grid.periodic) updateGhosts(level.v);
		
		Vector3 r = compResidualVec(grid, levelNum);
		
		for (std::size_t x = 0; x < level.levelDim[0] + 2; x++) {
			for (std::size_t y = 0; y < level.levelDim[1] + 2; y++) {
				for (std::size_t z = 0; z < level.levelDim[2] + 2; z++) {
					double newV = level.v.get(x, y, z) + grid.omega * (alpha * r.get(x, y, z));
					level.v.set(x, y, z, newV);
				}
			}
		}
	}

	if (grid.periodic) updateGhosts(level.v);
	return compResidual(grid, levelNum);
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

Vector3 CpuSolver::interpolate(const Vector3& src)
{
	std::size_t xDim = src.getXdim() - 2;
	std::size_t yDim = src.getYdim() - 2;
	std::size_t zDim = src.getZdim() - 2;

	xDim *= 2;
	yDim *= 2;
	zDim *= 2;

	xDim += 2;
	yDim += 2;
	zDim += 2;

	Vector3 dst(xDim, yDim, zDim);

	// prepare
	for (std::size_t x = 0; x < dst.getXdim() - 1; x += 2) {
		for (std::size_t y = 0; y < dst.getYdim() - 1; y += 2) {
			for (std::size_t z = 0; z < dst.getZdim() - 1; z += 2) {
				double val = src.get(x/2, y/2, z/2);
				dst.set(x, y, z, val);
			}
		}
	}

	// TODO: check for grid.periodic
	if (true) {
		updateGhosts(dst);
	}

	// Interpolate in x-direction
	for (std::size_t x = 0; x+2 < dst.getXdim(); x += 2) {
		for (std::size_t y = 0; y < dst.getYdim(); y += 2) {
			for (std::size_t z = 0; z < dst.getZdim(); z += 2) {
				double val = 0.5 * dst.get(x, y, z) + 0.5 * dst.get(x + 2, y, z);
				dst.set(x+1, y, z, val);
			}
		}
	}

	// Interpolate in y-direction
	for (std::size_t x = 0; x < dst.getXdim(); x++) {
		for (std::size_t y = 0; y + 2 < dst.getYdim(); y += 2) {
			for (std::size_t z = 0; z < dst.getZdim(); z += 2) {
				double val = 0.5 * dst.get(x, y, z) + 0.5 * dst.get(x, y+2, z);
				dst.set(x, y+1, z, val);
			}
		}
	}

	// Interpolate in z-direction
	for (std::size_t x = 0; x < dst.getXdim(); x++) {
		for (std::size_t y = 0; y < dst.getYdim(); y++) {
			for (std::size_t z = 0; z + 2 < dst.getZdim(); z += 2) {
				double val = 0.5 * dst.get(x, y, z) + 0.5 * dst.get(x, y, z + 2);
				dst.set(x, y, z+1, val);
			}
		}
	}

	// TODO: check for grid.periodic
	if (true) {
		updateGhosts(dst);
	}

	return dst;
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

	// y - dimension
	for (std::size_t x = 0; x < vec.getXdim(); x++) {
		for (std::size_t z = 0; z < vec.getZdim(); z++) {
			vec.set(x, 0, z, vec.get(x, vec.getYdim()-2, z));
			vec.set(x, vec.getYdim() - 1, z, vec.get(x, 1, z));
		}
	}

	for (std::size_t y = 0; y < vec.getYdim(); y++) {
		for (std::size_t z = 0; z < vec.getZdim(); z++) {
			vec.set(0, y, z, vec.get(vec.getXdim()-2, y, z));
			vec.set(vec.getXdim() - 1, y, z, vec.get(1, y, z));
		}
	}

}
