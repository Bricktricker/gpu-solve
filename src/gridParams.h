#pragma once
#include <array>
#include <vector>
#include <assert.h>
#include <tuple>

struct Stencil {
    std::vector<double> values;
    std::vector<std::tuple<int, int, int>> offsets;

    int getXOffset(std::size_t i) const
    {
        assert(i < offsets.size());
        return std::get<0>(offsets[i]);
    }
    int getYOffset(std::size_t i) const
    {
        assert(i < offsets.size());
        return std::get<1>(offsets[i]);
    }
    int getZOffset(std::size_t i) const
    {
        assert(i < offsets.size());
        return std::get<2>(offsets[i]);
    }

    static Stencil fromPrevLevel(const Stencil& prevStencil);
};

struct GridParams {
    std::size_t maxiter;
    double tol;
    double omega; // Relaxation coefficient
    double h;
    std::array<std::size_t, 3> gridDim;
    std::size_t preSmoothing;
    std::size_t postSmoothing;
    Stencil stencil{};
    bool periodic;
};
