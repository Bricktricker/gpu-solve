#pragma once
#include "../gridParams.h"
#include "SyclBuffer.h"
#include <CL/sycl.hpp>
#include <vector>
#include "sycl_compat.h"

class SyclGridData final : public GridParams
{
public:

	struct LevelData {
		SyclBuffer v;
		SyclBuffer restV;
		SyclBuffer f;
		SyclBuffer r;
		SyclBuffer e;
		std::array<std::size_t, 3> levelDim;
		Stencil stencil;
		double h;
	};

	SyclGridData(const GridParams& grid);

	void initBuffers(cl::sycl::queue& queue);

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
