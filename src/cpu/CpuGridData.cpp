#include "CpuGridData.h"
#include <math.h>
#include <algorithm>
#include <tuple>

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
	// magic 2.0 at the end is the coarsening ratio
	int maxlevel = (int)floor(log((double)std::min(std::min(gridDim[0], gridDim[1]), gridDim[2])) / log(2.0)) + 1;
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
		level.restV = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.newtonV = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.f = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		level.r = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		if (i + 1 != maxlevel) {
			level.e = Vector3(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);
		}

		level.h = 1.0 / (level.levelDim[1] + 1);
	}

	// fill right hand side for the first level
	if (this->mode == GridParams::LINEAR) {

		for (int i = 0; i < levels[0].levelDim[0]; i++) {
			for (int j = 0; j < levels[0].levelDim[1]; j++) {
				for (int k = 0; k < levels[0].levelDim[2]; k++) {
					double x = i * h;
					double y = j * h;
					double z = k * h;

					double val = -(f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));
					levels[0].f.set(i + 1, j + 1, k + 1, val);
				}
			}
		}

	}else{

		for (int i = 0; i < levels[0].levelDim[0] + 2; i++) {
			for (int j = 0; j < levels[0].levelDim[1] + 2; j++) {
				for (int k = 0; k < levels[0].levelDim[2] + 2; k++) {
					double x = i * h;
					double y = j * h;
					double z = k * h;

					double val = 2.0 * ((y - y * y) * (z - z * z) + (x - x * x) * (z - z * z) + (x - x * x) * (y - y * y))
						+ gamma * (x - x * x) * (y - y * y) * (z - z * z)
						* exp((x - x * x) * (y - y * y) * (z - z * z));

					levels[0].f.set(i, j, k, val);
				}
			}
		}

	}

}
