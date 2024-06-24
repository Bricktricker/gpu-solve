#include "CpuGridData.h"
#include <math.h>
#include <algorithm>
#include <tuple>

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
			level.stencil = stencil;
		}else {
			level.levelDim[0] = levels[i - 1].levelDim[0] / 2;
			level.levelDim[1] = levels[i - 1].levelDim[1] / 2;
			level.levelDim[2] = levels[i - 1].levelDim[2] / 2;
			level.stencil = Stencil::fromPrevLevel(levels[i - 1].stencil);
		}

		level.v = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.f = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.r = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		if (i + 1 != maxlevel) {
			level.e = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		}
	}

	// fill right hand side for the first level
	double sum = 0.0;
	for (int i = 0; i < levels[0].levelDim[0]; i++) {
		for (int j = 0; j < levels[0].levelDim[1]; j++) {
			for (int k = 0; k < levels[0].levelDim[2]; k++) {
				double x = i * h;
				double y = j * h;
				double z = k * h;

				double val = -h * h * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));
				levels[0].f.set(i+1, j+1, k+1, val);
				sum += val;
			}
		}
	}

	double left = sum / (levels[0].levelDim[0] * levels[0].levelDim[1] * levels[0].levelDim[2]);

	if (periodic) {
		for (int i = 0; i < levels[0].levelDim[0]; i++) {
			for (int j = 0; j < levels[0].levelDim[1]; j++) {
				for (int k = 0; k < levels[0].levelDim[2]; k++) {
					double newF = levels[0].f.get(i+1, j+1, k+1);
					newF -= left;
					levels[0].f.set(i+1, j+1, k+1, newF);
				}
			}
		}
	}

}
