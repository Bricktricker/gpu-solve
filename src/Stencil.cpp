#include "gridParams.h"
#include "cpu/Vector3.h"
#include <algorithm>
#include <assert.h>
#include <math.h>

namespace {
Vector3 conv3(const Vector3& a, const Vector3& b)
{
	int Aentriesx = static_cast<int>((a.getXdim() - 1) / 2);
	int Aentriesy = static_cast<int>((a.getYdim() - 1) / 2);
	int Aentriesz = static_cast<int>((a.getZdim() - 1) / 2);
	int Bentriesx = static_cast<int>((b.getXdim() - 1) / 2);
	int Bentriesy = static_cast<int>((b.getYdim() - 1) / 2);
	int Bentriesz = static_cast<int>((b.getZdim() - 1) / 2);

	int Centriesx = Aentriesx + Bentriesx;
	int Centriesy = Aentriesy + Bentriesy;
	int Centriesz = Aentriesz + Bentriesz;

	Vector3 c(2 * Centriesx + 1, 2 * Centriesy + 1, 2 * Centriesz + 1);

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

	return c;
}
}

/* TODO: Implement later?
Stencil Stencil::galerkin(const Stencil& prevStencil)
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
	Vector3 tmp = conv3(a, p);

	// calculate full coarse grid stencil
	Vector3 acFull = conv3(p, tmp);

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
					finalStencil.offsets[cnt] = std::make_tuple(int(i - 1), int(j - 1), int(k - 1));
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
*/

Stencil Stencil::simpleStencil(const Stencil& rootStencil, std::size_t level)
{
	assert(level > 0);
	double factor = 1.0;//pow(0.5, static_cast<double>(level));
	Stencil stencil = rootStencil;
	std::transform(stencil.values.begin(), stencil.values.end(), stencil.values.begin(), [&](double val) {
		return val / factor;
	});

	return stencil;
}
