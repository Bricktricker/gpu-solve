#pragma once
#include <array>
#include <vector>
#include <assert.h>
#include <tuple>

struct Stencil {
    std::array<double, 7> values;
    std::array<std::tuple<int, int, int>, 7> offsets;

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

    //static Stencil galerkin(const Stencil& prevStencil);
    static Stencil simpleStencil(const Stencil& rootStencil, std::size_t level);
};

struct GridParams {
    std::size_t maxiter;
    double tol;
    double omega; // Relaxation coefficient
    double gamma; // non-linear weight
    double h;
    std::array<std::size_t, 3> gridDim;
    std::size_t preSmoothing;
    std::size_t postSmoothing;
    Stencil stencil{};
    bool periodic;
};
