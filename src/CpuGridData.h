#pragma once
#include "gridParams.h"
#include <vector>

class CpuGridData final : public GridParams {
public:

    struct LevelData {
        std::vector<double> v; // left side, target
        std::vector<double> f; // right hand side

        std::array<std::size_t, 3> levelDim;

        double getF(std::size_t x, std::size_t y, std::size_t z) const;
        void setF(std::size_t x, std::size_t y, std::size_t z, double val);

        double getV(std::size_t x, std::size_t y, std::size_t z) const;
        void setV(std::size_t x, std::size_t y, std::size_t z, double val);
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
