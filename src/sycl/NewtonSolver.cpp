#include "NewtonSolver.h"
#include "SyclSolver.h"
#include "../Timer.h"
#include <fstream>
#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#endif

using namespace cl::sycl;

#ifndef SYCL_GTX
// Mark Stencil as device copyable
template<>
struct sycl::is_device_copyable<Stencil> : std::true_type {};
#endif

void NewtonSolver::solve(cl::sycl::queue& queue, SyclGridData& grid) {
    // newtonF already filled at this point
 
	// Compute inital residual
    double initialResidual = compF(queue, grid);
	std::cout << "Inital newton residual: " << initialResidual << '\n';

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();

        compF(queue, grid);
        // clear v
        queue.submit([&](handler& cgh) {
            SyclBuffer& v = grid.getLevel(0).v;
            auto vAcc = v.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class resetN>(range<1>(v.flatSize()), [=](id<1> index) {
                vAcc[index] = 0.0;
            });
        });

        findError(queue, grid);

        double res = compF(queue, grid);

        std::cout << "Newton iter: " << i << " residual: " << res << ' ';
		Timer::stop();

#ifdef _WIN32
        ::PROCESS_MEMORY_COUNTERS pmc = {};
        if (::GetProcessMemoryInfo(::GetCurrentProcess(), &pmc, sizeof(pmc))) {
            std::cout << "Current ram usage: " << pmc.WorkingSetSize << '\n';
        }
#endif

        if (res <= initialResidual / (1.0 / grid.tol)) {
            return;
        }

	}

}

double NewtonSolver::compF(cl::sycl::queue& queue, SyclGridData& grid)
{
    SyclGridData::LevelData& level = grid.getLevel(0);

    queue.submit([&](handler& cgh) {

        range<3> range(level.levelDim[0], level.levelDim[1], level.levelDim[2]);

        auto newtonfAcc = grid.newtonF.get_access<access::mode::read>(cgh);
        auto vAcc = level.newtonV.get_access<access::mode::read>(cgh);
        auto fAcc = level.f.get_access<access::mode::write>(cgh);

        cgh.parallel_for<class newtonF>(range, [=, h=level.h, gamma=grid.gamma, dims=level.f.getDims(), stencil=grid.stencil](id<3> index) {
            double1 stencilsum = 0.0;
            for (std::size_t i = 0; i < stencil.values.size(); i++) {
                const int1 flatIdx = Sycl3dAccesor::flatIndex(dims, index[0] + (stencil.getXOffset(i) + 1), index[1] + (stencil.getYOffset(i) + 1), index[2] + (stencil.getZOffset(i) + 1));
                auto vVal = vAcc[flatIdx];
                stencilsum += stencil.values[i] * vVal;
            }

            int1 centerIdx = Sycl3dAccesor::shift1Index(dims, index);
            stencilsum /= h * h;

            // See tutorial_multigrid.pdf, page 102, Formula 6.13
            double1 vVal = vAcc[centerIdx];
            double1 ex = cl::sycl::exp(vVal);
            double1 nonLinear = gamma * vVal * ex;
            stencilsum += nonLinear;

            // Can't combine them, generates false results
            double1 minus = newtonfAcc[centerIdx] - stencilsum;
            fAcc[centerIdx] = minus;
        });
    });

    return SyclSolver::sumBuffer(queue, level.f);
}

void NewtonSolver::findError(cl::sycl::queue& queue, SyclGridData& grid)
{
    SyclGridData mgGrid = grid;
    mgGrid.printProgress = false;
    mgGrid.maxiter = 10;
    mgGrid.tol = 0.1;

    for (std::size_t i = 1; i < grid.numLevels() - 1; i++) {
        SyclBuffer& src = mgGrid.getLevel(i - 1).newtonV;
        SyclBuffer& dst = mgGrid.getLevel(i).newtonV;
        SyclSolver::restrict(queue, src, dst);
    }

    SyclSolver::solve(queue, mgGrid);

    queue.submit([&](handler& cgh) {
        auto newtonvAcc = grid.getLevel(0).newtonV.get_access<access::mode::read_write>(cgh);
        auto vAcc = mgGrid.getLevel(0).v.get_access<access::mode::read>(cgh);

        cgh.parallel_for<class sumN>(range<1>(grid.getLevel(0).newtonV.flatSize()), [newtonvAcc, vAcc](id<1> index) {
            newtonvAcc[index] += vAcc[index];
        });
    });

}
