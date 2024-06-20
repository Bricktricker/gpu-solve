#include "SyclSolver.h"

// Usage: get an accessor for a buffer with host_buffer target
float getValueCPU(cl::sycl::accessor<float, 3, cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>& acc, std::size_t x, std::size_t y, std::size_t z)
{
    cl::sycl::detail::accessor_host_ref<1, float, 3, cl::sycl::access::mode::read> lastHostRef(acc.parent, std::array<std::size_t, 3>{x, y, 0});
    return lastHostRef[z];
}

template<class point_ref_x, class point_ref_y, class point_ref_z>
cl::sycl::detail::data_ref getValueGpu(
    const cl::sycl::accessor<float, 3, cl::sycl::access::mode::read>& acc,
    point_ref_x x, //cl::sycl::detail::point_ref<false>
    point_ref_y y,
    point_ref_z z
)
{
    auto xx = acc[x];
    auto yy = xx[y];
    auto tmp = *reinterpret_cast<cl::sycl::detail::accessor_device_ref<1, float, 3, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>*>(&yy);
    return tmp[z];
}

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

        cl::sycl::buffer<float, 3> buf3(cl::sycl::range<3>(10, 20, 5));

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
        assert(false); // TODO: shift index by 1, to get ro range (1,1,1) -> (dimX+1, dimY+1,dimZ+1)
        cl::sycl::float1 stencilsum = 6.0f * vAcc[index];
        stencilsum += -1.f * getValueGpu(vAcc, index[0] - 1, index[1], index[2]); // left
        stencilsum += -1.f * getValueGpu(vAcc, index[0] + 1, index[1], index[2]); // right
        stencilsum += -1.f * getValueGpu(vAcc, index[0], index[1] - 1, index[2]); // bottom
        stencilsum += -1.f * getValueGpu(vAcc, index[0], index[1] + 1, index[2]); // top
        stencilsum += -1.f * getValueGpu(vAcc, index[0], index[1], index[2] - 1); // front
        stencilsum += -1.f * getValueGpu(vAcc, index[0], index[1], index[2] + 1); // back

        rAcc[index] = fAcc[index] - stencilsum;
    });
}
