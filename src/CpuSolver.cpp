#include "CpuSolver.h"

void CpuSolver::solve(CpuGridData& grid)
{
	// Compute inital residual
	double initialResidual = compResidual(grid, 0);
}

double CpuSolver::compResidual(const CpuGridData& grid, std::size_t levelNum)
{
	double res = 0.0;
	const CpuGridData::LevelData& level = grid.getLevel(levelNum);

	for (std::size_t x = 1; x < level.levelDim[0]+1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1]+1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2]+1; z++) {
				
				// TODO: chnage order to reduce cache misses?
				double stencilsum = 0.0;
				stencilsum += grid.stencil.names.center * level.v.get(x, y, z);
				stencilsum += grid.stencil.names.left * level.v.get(x-1, y, z);
				stencilsum += grid.stencil.names.right * level.v.get(x + 1, y, z);
				stencilsum += grid.stencil.names.bottom * level.v.get(x, y-1, z);
				stencilsum += grid.stencil.names.top * level.v.get(x, y + 1, z);
				stencilsum += grid.stencil.names.front * level.v.get(x, y, z-1);
				stencilsum += grid.stencil.names.back * level.v.get(x, y, z + 1);

				double r = level.f.get(x - 1, y - 1, z - 1) - stencilsum;
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

				// TODO: chnage order to reduce cache misses?
				double stencilsum = 0.0;
				stencilsum += grid.stencil.names.center * level.v.get(x, y, z);
				stencilsum += grid.stencil.names.left * level.v.get(x - 1, y, z);
				stencilsum += grid.stencil.names.right * level.v.get(x + 1, y, z);
				stencilsum += grid.stencil.names.bottom * level.v.get(x, y - 1, z);
				stencilsum += grid.stencil.names.top * level.v.get(x, y + 1, z);
				stencilsum += grid.stencil.names.front * level.v.get(x, y, z - 1);
				stencilsum += grid.stencil.names.back * level.v.get(x, y, z + 1);

				double rVal = level.f.get(x - 1, y - 1, z - 1) - stencilsum;
				r.set(x - 1, y - 1, z - 1, rVal);
			}
		}
	}
	
	return r;
}



double CpuSolver::vcycle(const CpuGridData& grid)
{
	
	
	
	
	return 0.0; // returns current residual
}

double CpuSolver::jacobi(CpuGridData& grid, std::size_t level, std::size_t maxiter)
{
	for (std::size_t i = 0; i < maxiter; i++) {

	}
	
	return 0.0;
}
