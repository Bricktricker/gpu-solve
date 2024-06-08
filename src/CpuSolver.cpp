#include "CpuSolver.h"
#include <assert.h>
#include <iostream>

void CpuSolver::solve(CpuGridData& grid)
{
	// Compute inital residual
	double initialResidual = compResidual(grid, 0);

	double vReidual = vcycle(grid);

	int frzu = 0; // debug break point
}

double CpuSolver::compResidual(const CpuGridData& grid, std::size_t levelNum)
{
	double res = 0.0;
	const CpuGridData::LevelData& level = grid.getLevel(levelNum);

	for (std::size_t x = 1; x < level.levelDim[0]+1; x++) {
		for (std::size_t y = 1; y < level.levelDim[1]+1; y++) {
			for (std::size_t z = 1; z < level.levelDim[2]+1; z++) {
				
				// TODO: change order to reduce cache misses?
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

				// TODO: change order to reduce cache misses?
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



double CpuSolver::vcycle(CpuGridData& grid)
{
	for (std::size_t i = 0; i < grid.numLevels()-1; i++) {
		jacobi(grid, i, grid.preSmoothing);
		std::cout << "jacobi(" << i << ")\n";

		// clear v for next level
		CpuGridData::LevelData& nextLevel = grid.getLevel(i + 1);
		nextLevel.v.fill(0.0);
		std::cout << "v" << (i + 1) << " = 0 // clear old v\n";

		// compute residual
		Vector3 r = compResidualVec(grid, i);

		// restrict residual to next level f
		std::cout << "f" << (i + 1) << " = restrict(r" << i << ")\n";
		restrict(r, nextLevel.f);
	}
	
	// reached coarsed level, solve now
	std::cout << "jacobi(" << grid.numLevels() - 1 << ")\n";
	jacobi(grid, grid.numLevels() - 1, grid.postSmoothing);

	for (std::size_t i = grid.numLevels() - 1; i > 0; i--) {
		// interpolate v to previos level e
		std::cout << "e" << (i - 1) << " = interpolate(v" << i << ")\n";
		Vector3 e = interpolate(grid.getLevel(i).v);

		// v = v + e
		std::cout << 'v' << i - 1 << " += e" << i - 1 << '\n';
		Vector3& v = grid.getLevel(i - 1).v;
		v += e;

		std::cout << "jacobi(" << i - 1 << ")\n";
		jacobi(grid, i - 1, grid.postSmoothing);
	}

	// returns current residual
	return compResidual(grid, 0);
}

double CpuSolver::jacobi(CpuGridData& grid, std::size_t levelNum, std::size_t maxiter)
{
	const double alpha = 1.0 / grid.stencil.names.center;
	CpuGridData::LevelData& level = grid.getLevel(levelNum);
	
	for (std::size_t i = 0; i < maxiter; i++) {
		Vector3 r = compResidualVec(grid, levelNum);
		
		for (std::size_t x = 1; x < level.levelDim[0] + 1; x++) {
			for (std::size_t y = 1; y < level.levelDim[1] + 1; y++) {
				for (std::size_t z = 1; z < level.levelDim[2] + 1; z++) {
					double newV = level.v.get(x, y, z) + grid.omega * (alpha * r.get(x - 1, y - 1, z - 1));
					level.v.set(x, y, z, newV);
				}
			}
		}

	}

	return compResidual(grid, levelNum);
}

void CpuSolver::restrict(const Vector3& src, Vector3& dst)
{
	assert(src.flatSize() / 8 == dst.flatSize());

	// Previously the three loops run over  src and have only set the "middle" value, resulted in a better result, but why?

	for (std::size_t x = 0; x < dst.getXdim(); x++) {
		for (std::size_t y = 0; y < dst.getYdim(); y++) {
			for (std::size_t z = 0; z < dst.getZdim(); z++) {

				double middle = src.get(x*2, y*2, z*2);

				std::size_t borderValues = 0;
				double val = 0.0;

				if (x > 0) {
					val += src.get(x * 2 - 1, y * 2, z * 2); // left
					borderValues++;
				}
				if (y > 0) {
					val += src.get(x * 2, y * 2 - 1, z * 2); // bottom
					borderValues++;
				}
				if (z > 0) {
					val += src.get(x * 2, y * 2, z * 2 - 1); // front
					borderValues++;
				}

				if (x < dst.getXdim() - 1) {
					val += src.get(x * 2 + 1, y * 2, z * 2); // right
					borderValues++;
				}
				if (y < dst.getYdim() - 1) {
					val += src.get(x * 2, y * 2 + 1, z * 2); // top
					borderValues++;
				}
				if (z < dst.getZdim() - 1) {
					val += src.get(x * 2, y * 2, z * 2 + 1); // back
					borderValues++;
				}

				if (borderValues > 0) {
					val += borderValues * middle;
					val /= borderValues * 2;
				}

				dst.set(x / 2, y / 2, z / 2, val);
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

	for (std::size_t x = 1; x < dst.getXdim()-1; x++) {
		for (std::size_t y = 1; y < dst.getYdim()-1; y++) {
			for (std::size_t z = 1; z < dst.getZdim()-1; z++) {
				bool xEven = x % 2 == 0;
				bool yEven = y % 2 == 0;
				bool zEven = z % 2 == 0;

				if (xEven && yEven && zEven) {
					double val = src.get(x / 2, y / 2, z / 2);
					dst.set(x, y, z, val);
				}
				else {
					// TODO: correct interpolation? Maybe wrong when x is even
					double xVal = src.get(x / 2, y / 2, z / 2) + src.get((x / 2) + 1, y / 2, z / 2);
					double yVal = src.get(x / 2, y / 2, z / 2) + src.get(x/2, (y / 2) + 1, z / 2);
					double zVal = src.get(x / 2, y / 2, z / 2) + src.get(x/2, y / 2, (z / 2) + 1);
					dst.set(x, y, z, (xVal + yVal + zVal) / 6.0);
				}
			}
		}
	}

	return dst;
}
