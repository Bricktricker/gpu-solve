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
        queue.submit([&](cl::sycl::handler& cgh) {
            grid.initBuffers(cgh);
        });
    }
    
}
