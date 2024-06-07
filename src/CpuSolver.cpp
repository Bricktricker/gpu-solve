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
				stencilsum += grid.stencil.names.center * level.getV(x, y, z);
				stencilsum += grid.stencil.names.left * level.getV(x-1, y, z);
				stencilsum += grid.stencil.names.right * level.getV(x + 1, y, z);
				stencilsum += grid.stencil.names.bottom * level.getV(x, y-1, z);
				stencilsum += grid.stencil.names.top * level.getV(x, y + 1, z);
				stencilsum += grid.stencil.names.front * level.getV(x, y, z-1);
				stencilsum += grid.stencil.names.back * level.getV(x, y, z + 1);

				double r = level.getF(x - 1, y - 1, z - 1) - stencilsum;
				res += r * r;
			}
		}
	}
	
	return sqrt(res);
}

// like compResidual, but returns the vector r
std::vector<double> CpuSolver::compResidualVec(const CpuGridData& grid, std::size_t levelNum)
{
	const CpuGridData::LevelData& level = grid.getLevel(levelNum);

	std::vector<double> r(level.f.size());

	for (std::size_t x = 1; x < level.levelDim[0] + 1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1] + 1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2] + 1; z++) {

				// TODO: chnage order to reduce cache misses?
				double stencilsum = 0.0;
				stencilsum += grid.stencil.names.center * level.getV(x, y, z);
				stencilsum += grid.stencil.names.left * level.getV(x - 1, y, z);
				stencilsum += grid.stencil.names.right * level.getV(x + 1, y, z);
				stencilsum += grid.stencil.names.bottom * level.getV(x, y - 1, z);
				stencilsum += grid.stencil.names.top * level.getV(x, y + 1, z);
				stencilsum += grid.stencil.names.front * level.getV(x, y, z - 1);
				stencilsum += grid.stencil.names.back * level.getV(x, y, z + 1);

				double rVal = level.getF(x - 1, y - 1, z - 1) - stencilsum;
				
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
