#include "CpuGridData.h"
#include <math.h>
#include <assert.h>

// TODO: move them somewhere else
namespace {
double f0(double x) {
	return(100 * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0));
}
double f2(double x) {
	return(100.0 * 4.0 * (x - 1.0) * (x - 1.0) * x * x * (14.0 * x * x - 14.0 * x + 3));
}
}

void CpuGridData::LevelData::setF(std::size_t x, std::size_t y, std::size_t z, double val)
{
	const std::size_t idx = (z * levelDim[0] * levelDim[1]) + (y * levelDim[0]) + x;
	assert(idx < f.size());
	f[idx] = val;
}

double CpuGridData::LevelData::getF(std::size_t x, std::size_t y, std::size_t z) const
{
	const std::size_t idx = (z * levelDim[0] * levelDim[1]) + (y * levelDim[0]) + x;
	assert(idx < f.size());
	return f[idx];
}

double CpuGridData::LevelData::getV(std::size_t x, std::size_t y, std::size_t z) const
{
	std::size_t xMax = levelDim[0] + 2;
	std::size_t yMax = levelDim[1] + 2;
	const std::size_t idx = (z * xMax * yMax) + (y * xMax) + x;
	assert(idx < v.size());
	return v[idx];
}

void CpuGridData::LevelData::setV(std::size_t x, std::size_t y, std::size_t z, double val)
{
	std::size_t xMax = levelDim[0] + 2;
	std::size_t yMax = levelDim[1] + 2;
	const std::size_t idx = (z * xMax * yMax) + (y * xMax) + x;
	assert(idx < v.size());
	v[idx] = val;
}

CpuGridData::CpuGridData(const GridParams& grid)
	: GridParams(grid)
{
	int maxlevel;
	// magic 2.0 at the end is the coarsening ratio
	if (periodic) {
		maxlevel = (int)floor(log((double)(std::min(std::min(gridDim[0], gridDim[1]), gridDim[2]) + 1)) / log(2.0)) + 1;
	}else {
		maxlevel = (int)floor(log((double)std::min(std::min(gridDim[0], gridDim[1]), gridDim[2])) / log(2.0)) + 1;
	}
	levels.resize(maxlevel);

	for (std::size_t i = 0; i < levels.size(); i++) {
		auto& level = levels[i];
		if (i == 0) {
			level.levelDim = gridDim;
		}else {
			level.levelDim[0] = levels[i - 1].levelDim[0] / 2;
			level.levelDim[1] = levels[i - 1].levelDim[1] / 2;
			level.levelDim[2] = levels[i - 1].levelDim[2] / 2;
		}

		const std::size_t numValuesLeft = (level.levelDim[0] + 2) * (level.levelDim[1] + 2) * (level.levelDim[2] + 2);
		const std::size_t numValuesRight = level.levelDim[0] * level.levelDim[1] * level.levelDim[2];

		level.v.resize(numValuesLeft);
		level.f.resize(numValuesRight);
	}

	// fill right hand side for the first level
	for (std::size_t i = 0; i < levels[0].levelDim[0]; i++) {
		for (std::size_t j = 0; j < levels[0].levelDim[1]; j++) {
			for (std::size_t k = 0; k < levels[0].levelDim[2]; k++) {
				double x = i * h;
				double y = j * h;
				double z = k * h;

				double val = -h * h * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));
				levels[0].setF(i, j, k, val); // TODO: change function or iteration order to not jump around in memory
			}
		}
	}
}