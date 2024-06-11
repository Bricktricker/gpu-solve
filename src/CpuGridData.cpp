#include "CpuGridData.h"
#include <math.h>

// TODO: move them somewhere else
namespace {
double f0(double x) {
	return(100 * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0));
}
double f2(double x) {
	return(100.0 * 4.0 * (x - 1.0) * (x - 1.0) * x * x * (14.0 * x * x - 14.0 * x + 3));
}
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

		level.v = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.f = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
	}

	// fill right hand side for the first level
	double sum = 0.0;
	for (int i = 0; i < levels[0].levelDim[0]; i++) {
		for (int j = 0; j < levels[0].levelDim[1]; j++) {
			for (int k = 0; k < levels[0].levelDim[2]; k++) {
				double x = (i-0) * h;
				double y = (j-0) * h;
				double z = (k-0) * h;

				double val = -h * h * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));
				levels[0].f.set(i, j, k, val); // TODO: change function or iteration order to not jump around in memory
				sum += val;
			}
		}
	}

	double left = sum / (levels[0].levelDim[0] * levels[0].levelDim[1] * levels[0].levelDim[2]);
	printf("sum: %.17g, left: %.17g\n", sum, left);

	if (periodic) {
		for (int i = 0; i < levels[0].levelDim[0]; i++) {
			for (int j = 0; j < levels[0].levelDim[1]; j++) {
				for (int k = 0; k < levels[0].levelDim[2]; k++) {
					double newF = levels[0].f.get(i, j, k);
					newF -= left;
					levels[0].f.set(i, j, k, newF);
				}
			}
		}
	}

	// Move all values in levels[0].f one to the right
	// TODO: change code above so we don't need a tmp vector
	Vector3 tmp(levels[0].levelDim[0] + 2, levels[0].levelDim[1] + 2, levels[0].levelDim[2] + 2);
	for (int i = 0; i < tmp.getXdim()-1; i++) {
		for (int j = 0; j < tmp.getYdim()-1; j++) {
			for (int k = 0; k < tmp.getZdim()-1; k++) {
				double val = levels[0].f.get(i, j, k);
				tmp.set(i+1, j+1, k+1, val);
			}
		}
	}
	levels[0].f = tmp;
}
