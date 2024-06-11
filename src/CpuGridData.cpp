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
			level.stencil = stencil;
		}else {
			level.levelDim[0] = levels[i - 1].levelDim[0] / 2;
			level.levelDim[1] = levels[i - 1].levelDim[1] / 2;
			level.levelDim[2] = levels[i - 1].levelDim[2] / 2;
			level.stencil = computeStencil(levels[i-1].stencil);
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

Stencil CpuGridData::computeStencil(const Stencil& prevStencil) const
{
	// create stencil for prolongation
	Vector3 p(3, 3, 3);
	for (int i = -2 + 1; i < 2; i++) {
		for (int j = -2 + 1; j < 2; j++) {
			for (int k = -2 + 1; k < 2; k++) {
				double val = (2 - abs(i)) * (2 - abs(j)) * (2 - abs(k)) / ((double)(2 * 2 * 2));
				p.set(i + 2 - 1, j + 2 - 1, k + 2 - 1, val);
			}
		}
	}

	// create stencil for fine grid operator
	Vector3 a(3, 3, 3);
	for (int i = 0; i < prevStencil.values.size(); i++) {
		a.set(prevStencil.getXOffset(i) + 1, prevStencil.getYOffset(i) + 1, prevStencil.getZOffset(i) + 1, prevStencil.values[i]);
	}

	// calculate temporary stencil
	Vector3 tmp(5, 5, 5);
	conv3(a, p, tmp);

	// calculate full coarse grid stencil
	Vector3 acFull(7, 7, 7);
	conv3(p, tmp, acFull);

	// cut coarse grid stencil
	Vector3 ac(3, 3, 3);
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				double val = acFull.get(2 * i + 3, 2 * j + 3, 2 * k + 3);
				ac.set(i + 1, j + 1, k + 1, val);
			}
		}
	}

	// 3 x 3 x 3
	Stencil finalStencil;
	finalStencil.values.resize(27);
	finalStencil.offsets.resize(27);

	std::size_t cnt = 0;
	for (std::size_t i = 0; i < ac.getXdim(); i++)
		for (std::size_t j = 0; j < ac.getYdim(); j++)
			for (std::size_t k = 0; k < ac.getZdim(); k++)
				if (ac.get(i, j, k) != 0.0) {
					finalStencil.values[cnt] = ac.get(i, j, k);
					finalStencil.offsets[cnt] = std::make_tuple(i - 1, j - 1, k - 1);
					cnt++;
				}

	// sort stencil
	for (std::size_t i = 0; i < finalStencil.values.size(); i++) {
		if (finalStencil.getXOffset(i) == 0 && finalStencil.getYOffset(i) == 0 && finalStencil.getZOffset(i) == 0) {
			double tmp_value = finalStencil.values[0];
			auto tmpOffset = finalStencil.offsets[0];
			finalStencil.values[0] = finalStencil.values[i];
			finalStencil.offsets[0] = finalStencil.offsets[i];
			finalStencil.values[i] = tmp_value;
			finalStencil.offsets[i] = tmpOffset;
		}
	}
	
	return finalStencil;
}

void CpuGridData::conv3(const Vector3& a, const Vector3& b, Vector3& c)
{
	int Aentriesx = (a.getXdim() - 1) / 2;
	int Aentriesy = (a.getYdim() - 1) / 2;
	int Aentriesz = (a.getZdim() - 1) / 2;
	int Bentriesx = (b.getXdim() - 1) / 2;
	int Bentriesy = (b.getYdim() - 1) / 2;
	int Bentriesz = (b.getZdim() - 1) / 2;

	int Centriesx = Aentriesx + Bentriesx;
	int Centriesy = Aentriesy + Bentriesy;
	int Centriesz = Aentriesz + Bentriesz;

	if (c.getXdim() != 2 * Centriesx + 1) printf("Error! Size of array C does not match!\n");
	if (c.getYdim() != 2 * Centriesy + 1) printf("Error! Size of array C does not match!\n");
	if (c.getZdim() != 2 * Centriesz + 1) printf("Error! Size of array C does not match!\n");

	for (int i = -Centriesx; i <= Centriesx; i++)
		for (int j = -Centriesy; j <= Centriesy; j++)
			for (int k = -Centriesz; k <= Centriesz; k++)
				for (int ii = std::max(-Aentriesx, i - Bentriesx); ii <= std::min(Aentriesx, i + Bentriesx); ii++)
					for (int jj = std::max(-Aentriesy, j - Bentriesy); jj <= std::min(Aentriesy, j + Bentriesy); jj++)
						for (int kk = std::max(-Aentriesz, k - Bentriesz); kk <= std::min(Aentriesz, k + Bentriesz); kk++) {
							double val = c.get(i + Centriesx, j + Centriesy, k + Centriesz) +
								b.get(ii - i + Bentriesx, jj - j + Bentriesy, kk - k + Bentriesz) *
								a.get(ii + Aentriesx, jj + Aentriesy, kk + Aentriesz);
							c.set(i + Centriesx, j + Centriesy, k + Centriesz, val);
						}
}
