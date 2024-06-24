#include "SyclGridData.h"

// TODO: move them somewhere else
namespace {
	template<class Float>
	Float f0(Float x) {
		return Float(100 * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0) * x * (x - 1.0));
	}
	template<class Float>
	Float f2(Float x) {
		return Float(100.0 * 4.0 * (x - 1.0) * (x - 1.0) * x * x * (14.0 * x * x - 14.0 * x + 3));
	}
}

SyclGridData::SyclGridData(const GridParams& grid)
	: GridParams(grid)
{
	int maxlevel;
	// magic 2.0 at the end is the coarsening ratio
	if (periodic) {
		maxlevel = (int)floor(log((double)(std::min(std::min(gridDim[0], gridDim[1]), gridDim[2]) + 1)) / log(2.0)) + 1;
	}
	else {
		maxlevel = (int)floor(log((double)std::min(std::min(gridDim[0], gridDim[1]), gridDim[2])) / log(2.0)) + 1;
	}
	levels.reserve(maxlevel);

	for (std::size_t i = 0; i < maxlevel; i++) {
		std::array<std::size_t, 3> levelDim;
		Stencil levelStencil;
		if (i == 0) {
			levelDim = gridDim;
			levelStencil = stencil;
		}else {
			levelDim[0] = levels[i - 1].levelDim[0] / 2;
			levelDim[1] = levels[i - 1].levelDim[1] / 2;
			levelDim[2] = levels[i - 1].levelDim[2] / 2;
			levelStencil = Stencil::fromPrevLevel(levels[i - 1].stencil);
		}

		levels.push_back(LevelData{
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			SyclBuffer(levelDim[0] + 2, levelDim[1] + 2, levelDim[2] + 2),
			levelDim,
			levelStencil
		});

	}
}

void SyclGridData::initBuffers(cl::sycl::handler& cgh)
{
	// Compute right hand side sum, if we are periodic
	double leftFac = 0.0f;
	if (this->periodic) {
		double sum = 0.0;
		for (int i = 0; i < levels[0].levelDim[0]; i++) {
			for (int j = 0; j < levels[0].levelDim[1]; j++) {
				for (int k = 0; k < levels[0].levelDim[2]; k++) {
					double x = i * h;
					double y = j * h;
					double z = k * h;

					double val = -h * h * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));
					sum += val;
				}
			}
		}
		leftFac = sum / (levels[0].levelDim[0] * levels[0].levelDim[1] * levels[0].levelDim[2]);
	}
	
	auto wAccessor = levels[0].f.get_access<cl::sycl::access::mode::discard_write>(cgh);
	cl::sycl::range<3> range(levels[0].levelDim[0]+2, levels[0].levelDim[1]+2, levels[0].levelDim[2]+2);

	const auto xRightSide = levels[0].levelDim[0] + 1;
	const auto yRightSide = levels[0].levelDim[1] + 1;
	const auto zRightSide = levels[0].levelDim[2] + 1;

	cgh.parallel_for<class init_f>(range, [=](cl::sycl::id<3> index) {
		SYCL_IF(index[0] == 0 || index[1] == 0 || index[2] == 0) {
			wAccessor(index) = 0;
		}
		SYCL_ELSE_IF(index[0] == xRightSide || index[1] == yRightSide || index[2] == zRightSide) {
			wAccessor(index) = 0;
		}
		SYCL_ELSE
		{
			cl::sycl::double1 x = index[0] * h;
			cl::sycl::double1 y = index[1] * h;
			cl::sycl::double1 z = index[2] * h;

			cl::sycl::double1 val = (- h * h) * (f2(x)* f0(y)* f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));

			assert(leftFac == 0.0);
			wAccessor(index) = val; //-leftFac;
		}
		SYCL_END;
	});

	// Init other buffers to 0
	// TODO: Do I need to init all buffers? Can't I skip e and r?
	for (std::size_t i = 0; i < levels.size(); i++) {
		auto& level = levels[i];
		cl::sycl::range<3> bufferRange(level.levelDim[0] + 2, level.levelDim[1] + 2, level.levelDim[2] + 2);

		if (i > 0) {
			auto fAcc = level.f.get_access<cl::sycl::access::mode::discard_write>(cgh);
			cgh.parallel_for<class clear>(bufferRange, [=](cl::sycl::id<3> index) {
				fAcc(index) = 0.0;
			});
		}

		auto vAcc = level.v.get_access<cl::sycl::access::mode::discard_write>(cgh);
		auto rAcc = level.r.get_access<cl::sycl::access::mode::discard_write>(cgh);
		auto eAcc = level.e.get_access<cl::sycl::access::mode::discard_write>(cgh);
		cgh.parallel_for<class clear>(bufferRange, [=](cl::sycl::id<3> index) {
			vAcc(index) = 0.0;
			rAcc(index) = 0.0;
			eAcc(index) = 0.0;
		});
	}

}
