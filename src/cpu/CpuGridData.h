#pragma once
#include "../gridParams.h"
#include "Vector3.h"
#include <vector>

class CpuGridData final : public GridParams {
public:

    struct LevelData {
        Vector3 v; // left side, target
        Vector3 restV; // restricted v from previous level
        Vector3 newtonV; // v from the newton step
        Vector3 f; // right hand side
        Vector3 r; // latest residual
        Vector3 e; // error
        std::array<std::size_t, 3> levelDim;
        double h;
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

    std::size_t numLevels() const
    {
        return levels.size();
    }

private:
    std::vector<LevelData> levels; // levels[0] is the finest level and levels[-1] is the coarsed level
};
