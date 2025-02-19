#include "SyclGridData.h"

namespace {
	template<class Float>
	Float f0(const Float& x) {
		return (100 * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0));
	}
	template<class Float>
	Float f2(const Float& x) {
		return (100.0 * 4.0 * (x - 1.0) * (x - 1.0) * x * x * (14.0 * x * x - 14.0 * x + 3));
	}
}

SyclGridData::SyclGridData(const GridParams& grid)
	: GridParams(grid), newtonF(gridDim[0] + 2, gridDim[1] + 2, gridDim[2] + 2)
{
	// magic 2.0 at the end is the coarsening ratio
	int maxlevel = (int)floor(log((double)std::min(std::min(gridDim[0], gridDim[1]), gridDim[2])) / log(2.0)) + 1;
	levels.reserve(maxlevel);

	for (std::size_t i = 0; i < maxlevel; i++) {
		std::array<std::size_t, 3> levelDim;
		if (i == 0) {
			levelDim = gridDim;
		}else {
			levelDim[0] = levels[i - 1].levelDim[0] / 2;
			levelDim[1] = levels[i - 1].levelDim[1] / 2;
			levelDim[2] = levels[i - 1].levelDim[2] / 2;
		}

		double h = 1.0 / (levelDim[1] + 1);

		levels.push_back(LevelData{
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			levelDim,
			h
		});

	}
}

void SyclGridData::initBuffers(cl::sycl::queue& queue)
{	
	queue.submit([&, h=this->h](cl::sycl::handler& cgh) {
		auto wAccessor = levels[0].f.get_access<cl::sycl::access::mode::discard_write>(cgh);
		cl::sycl::range<3> range(levels[0].levelDim[0] + 2, levels[0].levelDim[1] + 2, levels[0].levelDim[2] + 2);

		const auto xRightSide = levels[0].levelDim[0] + 1;
		const auto yRightSide = levels[0].levelDim[1] + 1;
		const auto zRightSide = levels[0].levelDim[2] + 1;

		if (this->mode == GridParams::LINEAR) {

			cgh.parallel_for<class init_f_lin>(range, [=, h = this->h, dims = levels[0].f.getDims()](cl::sycl::id<3> index) {
				int1 flatIndex = Sycl3dAccesor::flatIndex(dims, index);
				SYCL_IF(index[0] == 0 || index[1] == 0 || index[2] == 0) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE_IF(index[0] == xRightSide || index[1] == yRightSide || index[2] == zRightSide) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE
				{
					double1 x = (index[0] - 1) * h;
					double1 y = (index[1] - 1) * h;
					double1 z = (index[2] - 1) * h;

					double1 val = -1.0 * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));

					wAccessor[flatIndex] = val;
				}
				SYCL_END;
			});
		}else {
			cgh.parallel_for<class init_f>(range, [=, h=this->h, ga=gamma, dims=levels[0].f.getDims()](cl::sycl::id<3> index) {
				int1 flatIndex = Sycl3dAccesor::flatIndex(dims, index);

				SYCL_IF(index[0] == 0 || index[1] == 0 || index[2] == 0) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE_IF(index[0] == xRightSide || index[1] == yRightSide || index[2] == zRightSide) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE
				{
					double1 x = index[0] * h;
					double1 y = index[1] * h;
					double1 z = index[2] * h;

					double1 val = 2.0 * ((y - y * y) * (z - z * z) + (x - x * x) * (z - z * z) + (x - x * x) * (y - y * y))
						+ ga * (x - x * x) * (y - y * y) * (z - z * z)
						* cl::sycl::exp((x - x * x) * (y - y * y) * (z - z * z));

					wAccessor[flatIndex] = val;
				}
				SYCL_END;
			});
		}
	});

	if (mode == GridParams::NEWTON) {
		queue.submit([&](cl::sycl::handler& cgh) {
			// Store the original right hand side in newtonF, never gets changed
			auto newtonfAcc = newtonF.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto fAcc = levels[0].f.get_access<cl::sycl::access::mode::read>(cgh);
			cgh.parallel_for<class init_newton>(cl::sycl::range<1>(newtonF.flatSize()), [newtonfAcc, fAcc](cl::sycl::id<1> index) {
				newtonfAcc[index] = fAcc[index];
			});
		});
	}

	// Init other buffers to 0
	// TODO: Do I need to init all buffers? Can't I skip e and r?
	for (std::size_t i = 0; i < levels.size(); i++) {
		auto& level = levels[i];

		if (i > 0) {
			queue.submit([&](cl::sycl::handler& cgh) {
				auto fAcc = level.f.get_access<cl::sycl::access::mode::discard_write>(cgh);
				cgh.parallel_for<class clear>(cl::sycl::range<1>(level.f.flatSize()), [=](cl::sycl::id<1> index) {
					fAcc[index] = 0.0;
				});
			});
		}

		queue.submit([&](cl::sycl::handler& cgh) {
			auto vAcc = level.v.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto restvAcc = level.restV.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto newtonvAcc = level.newtonV.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto rAcc = level.r.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto eAcc = level.e.get_access<cl::sycl::access::mode::discard_write>(cgh);
			cgh.parallel_for<class clearAll>(cl::sycl::range<1>(level.v.flatSize()), [vAcc, restvAcc, newtonvAcc, rAcc, eAcc](cl::sycl::id<1> index) {
				vAcc[index] = 0.0;
				restvAcc[index] = 0.0;
				newtonvAcc[index] = 0.0;
				rAcc[index] = 0.0;
				eAcc[index] = 0.0;
			});
		});
	}

}
