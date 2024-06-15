#pragma once
#include "gridParams.h"
#include <cl/sycl.hpp>
#include <vector>

class SyclGridData final : public GridParams
{
public:

	struct LevelData {
		cl::sycl::buffer<float, 3> v;
		cl::sycl::buffer<float, 3> f;
		cl::sycl::buffer<float, 3> r;
		cl::sycl::buffer<float, 3> e;
		std::array<std::size_t, 3> levelDim;
		// GpuStencil
	};

	SyclGridData(const GridParams& grid);

	void initBuffers(cl::sycl::handler& cgh);

	const LevelData& getLevel(std::size_t level) const
	{
		return levels[level];
	}
	LevelData& getLevel(std::size_t level)
	{
		return levels[level];
	}

	std::size_t numLevels() const
	{
		return levels.size();
	}

private:
	std::vector<LevelData> levels; // levels[0] is the finest level and levels[-1] is the coarsed level
};