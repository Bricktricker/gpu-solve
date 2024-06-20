#include "SyclSolver.h"

void SyclSolver::solve(SyclGridData& grid)
{
    auto platforms = cl::sycl::platform::get_platforms();
    std::cout << "Number of platforms: " << platforms.size() << '\n';
    cl::sycl::platform P = platforms.at(0);

    const auto devices = P.get_devices(cl::sycl::info::device_type::gpu);
    cl::sycl::device D = devices[0];
    
    {
        cl::sycl::queue queue(D);
        
        // TODO: move everything into one submit?
        queue.submit([&](cl::sycl::handler& cgh) {
            grid.initBuffers(cgh);
        });

        queue.submit([&](cl::sycl::handler& cgh) {
            compResidual(cgh, grid, 0);
        });
    }
    
}

void SyclSolver::vsycle(cl::sycl::handler& cgh, SyclGridData& grid)
{
    for (std::size_t i = 0; i < grid.numLevels() - 1; i++) {
        
    }
}

void SyclSolver::jacobi(cl::sycl::handler& cgh, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter)
{

}

void SyclSolver::compResidual(cl::sycl::handler& cgh, SyclGridData& grid, std::size_t levelNum)
{
    SyclGridData::LevelData& level = grid.getLevel(levelNum);

    cl::sycl::range<3> range(level.levelDim[0], level.levelDim[1], level.levelDim[2]);

    auto fAcc = level.f.get_access<cl::sycl::access::mode::read>(cgh);
    auto vAcc = level.v.get_access<cl::sycl::access::mode::read>(cgh);
    auto rAcc = level.r.get_access<cl::sycl::access::mode::write>(cgh);

    cgh.parallel_for<class res>(range, [=](cl::sycl::id<3> index) {
        cl::sycl::int1 centerIdx = vAcc.shift1Index(index);

        cl::sycl::float1 stencilsum = 6.0f * vAcc[centerIdx];
        stencilsum += -1.f * vAcc.shift1(index[0] - 1, index[1], index[2]); // left
        stencilsum += -1.f * vAcc.shift1(index[0] + 1, index[1], index[2]); // right
        stencilsum += -1.f * vAcc.shift1(index[0], index[1] - 1, index[2]); // bottom
        stencilsum += -1.f * vAcc.shift1(index[0], index[1] + 1, index[2]); // top
        stencilsum += -1.f * vAcc.shift1(index[0], index[1], index[2] - 1); // front
        stencilsum += -1.f * vAcc.shift1(index[0], index[1], index[2] + 1); // back

        rAcc[centerIdx] = fAcc[centerIdx] - stencilsum;
    });
}
