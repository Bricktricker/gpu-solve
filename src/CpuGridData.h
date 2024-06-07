#pragma once
#include "gridParams.h"
#include "Vector3.h"
#include <vector>

class CpuGridData final : public GridParams {
public:

    struct LevelData {
        Vector3 v; // left side, target
        Vector3 f; // right hand side
        std::array<std::size_t, 3> levelDim;
    };

    CpuGridData(const GridParams& grid);

    const LevelData& getLevel(std::size_t level) const
    {
        return levels[level];
    }
    LevelData& getLevel(std::size_t level)
    {
        return levels[level];
    }

private:
    std::vector<LevelData> levels; // levels[0] is the finest level and levels[-1] is the coarsed level
};
