#pragma once
#include <array>

// TODO: convert to normal struct?
union Stencil {
    std::array<double, 7> values;
    struct Names {
        double center, left, right, top, bottom, front, back;
    } names;
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
