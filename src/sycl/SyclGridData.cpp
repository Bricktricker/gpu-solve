#include "SyclGridData.h"

// TODO: move them somewhere else
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
			levelStencil = Stencil::simpleStencil(this->stencil, i);
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

void SyclGridData::initBuffers(cl::sycl::queue& queue)
{	
	queue.submit([&, h=this->h](cl::sycl::handler& cgh) {
		auto wAccessor = levels[0].f.get_access<cl::sycl::access::mode::discard_write>(cgh);
		cl::sycl::range<3> range(levels[0].levelDim[0] + 2, levels[0].levelDim[1] + 2, levels[0].levelDim[2] + 2);

		constexpr std::size_t work_group_size = 32;
		const std::size_t global_work_items = range.size();
		const std::size_t num_work_items = (global_work_items + work_group_size - 1); // ceil(global_work_items / work_group_size)
		cl::sycl::nd_range<1> nd_range(num_work_items, work_group_size);

		const auto xRightSide = levels[0].levelDim[0] + 1;
		const auto yRightSide = levels[0].levelDim[1] + 1;
		const auto zRightSide = levels[0].levelDim[2] + 1;

		cgh.parallel_for<class init_ff>(nd_range, [=, h=this->h, dims=levels[0].f.getDims()](cl::sycl::nd_item<1> index) {
			int1 flatIndex = index.get_global_id();
			SYCL_IF(flatIndex < global_work_items) {
				const int3 cubeIndex = Sycl3dAccesor::cubeIndex(dims, flatIndex);

				SYCL_IF(cubeIndex.x() == 0 || cubeIndex.y() == 0 || cubeIndex.z() == 0) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE_IF(cubeIndex.x() == xRightSide || cubeIndex.y() == yRightSide || cubeIndex.z() == zRightSide) {
					wAccessor[flatIndex] = 0.0;
				}
				SYCL_ELSE
				{
					double1 x = (cubeIndex.x() - 1) * h;
					double1 y = (cubeIndex.y() - 1) * h;
					double1 z = (cubeIndex.z() - 1) * h;

					double1 val = (-h * h) * (f2(x) * f0(y) * f0(z) + f0(x) * f2(y) * f0(z) + f0(x) * f0(y) * f2(z));

					wAccessor[flatIndex] = val;
				}
				SYCL_END;
			}
			SYCL_END;
		});
	});

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
			auto rAcc = level.r.get_access<cl::sycl::access::mode::discard_write>(cgh);
			auto eAcc = level.e.get_access<cl::sycl::access::mode::discard_write>(cgh);
			cgh.parallel_for<class clearAll>(cl::sycl::range<1>(level.v.flatSize()), [vAcc, rAcc, eAcc](cl::sycl::id<1> index) {
				vAcc[index] = 0.0;
				rAcc[index] = 0.0;
				eAcc[index] = 0.0;
			});
		});
	}

}
