#include "NewtonSolver.h"
#include "SyclSolver.h"
#include <fstream>

using namespace cl::sycl;

// Used for debugging to read values on the host
#ifdef SYCL_GTX
namespace {
    double getGpuVal(accessor<double, 1, access::mode::read, access::target::host_buffer>& acc, const SyclBuffer& buf, std::size_t x, std::size_t y, std::size_t z)
    {
        const std::size_t idx1 = z * (buf.getYdim() * buf.getXdim()) + y * buf.getXdim() + x;
        return acc[static_cast<int>(idx1)];
    }
    void dumpGpuBuf(SyclBuffer& buf, const std::string& file)
    {
        auto acc = buf.get_host_access<access::mode::read>();
        std::ofstream out;
        if (!file.empty()) {
            out.open(file);
        }

        for (std::size_t x = 0; x < buf.getXdim(); x++) {
            for (std::size_t y = 0; y < buf.getYdim(); y++) {
                for (std::size_t z = 0; z < buf.getZdim(); z++) {
                    if (out) {
                        out << "Index: " << x << ' ' << y << ' ' << z << " Value: " << getGpuVal(acc, buf, x, y, z) << '\n';
                    }
                    else {
                        std::cout << "Index: " << x << ' ' << y << ' ' << z << " Value: " << getGpuVal(acc, buf, x, y, z) << '\n';
                    }
                }
            }
        }
    }
}
#endif

void NewtonSolver::solve(ContextHandles& handles, SyclGridData& grid) {
	// Compute inital residual

	grid.mode = GridParams::Mode::NONLINEAR;
	double initialResidual = SyclSolver::sumResidual(handles.queue, grid, 0);
	std::cout << "Inital newton residual: " << initialResidual << '\n';
    grid.mode = GridParams::Mode::NEWTON;

	// newtonF already filled at this point
    //dumpGpuBuf(grid.newtonF, "newtonF.txt");
    //dumpGpuBuf(grid.getLevel(0).f, "f_before.txt");

	for (std::size_t i = 0; i < grid.maxiter; i++) {
		Timer::start();

		compF(handles.queue, grid);
        
        // clear v
        handles.queue.submit([&](handler& cgh) {
            SyclBuffer& v = grid.getLevel(0).v;
            auto vAcc = v.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class reset>(range<1>(v.flatSize()), [=](id<1> index) {
                vAcc[index] = 0.0;
            });
        });

        findError(handles, grid);

        // clear v
        handles.queue.submit([&](handler& cgh) {
            SyclBuffer& v = grid.getLevel(0).v;
            auto vAcc = v.get_access<access::mode::discard_write>(cgh);
            cgh.parallel_for<class reset>(range<1>(v.flatSize()), [=](id<1> index) {
                vAcc[index] = 0.0;
            });
        });
        compF(handles.queue, grid);
        grid.mode = GridParams::Mode::NONLINEAR;
        double res = SyclSolver::sumResidual(handles.queue, grid, 0);
        grid.mode = GridParams::Mode::NEWTON;

        std::cout << "Newton iter: " << i << " residual: " << res << ' ';
		Timer::stop();
	}

}

void NewtonSolver::compF(cl::sycl::queue& queue, SyclGridData& grid)
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

            fAcc[centerIdx] = newtonfAcc[centerIdx] - stencilsum;
        });
    });

    queue.wait();

    {
        const auto flatIndex = [](const BufferDim& dims, const int x, const int y, const int z) {
            return z * (dims[1] * dims[0]) + y * dims[0] + x;
        };
        
        auto newtonfAcc = grid.newtonF.get_host_access<access::mode::read>();
        auto vAcc = level.newtonV.get_host_access<access::mode::read>();
        auto fAcc = level.f.get_host_access<access::mode::write>();

        int x = 30;
        int y = 35;
        int z = 50;

        double stencilsum = 0.0;
        for (std::size_t i = 0; i < grid.stencil.values.size(); i++) {
            const int flatIdx = flatIndex(level.f.getDims(), x + (grid.stencil.getXOffset(i) + 1), y + (grid.stencil.getYOffset(i) + 1), z + (grid.stencil.getZOffset(i) + 1));
            auto vVal = vAcc[flatIdx];
            stencilsum += grid.stencil.values[i] * vVal;
        }
        stencilsum /= level.h * level.h;

        const auto& dims = level.f.getDims();

        int centerIdx = (z + 1) * (dims[1] * dims[0]) + (y + 1) * dims[0] + (x + 1);
        double vVal = vAcc[centerIdx];
        double ex = exp(vVal);
        double nonLinear = grid.gamma * vVal * ex;
        stencilsum += nonLinear;

        double minus = newtonfAcc[centerIdx] - stencilsum;
        fAcc[centerIdx] = minus;
    }

    dumpGpuBuf(level.f, "f.txt");

    double fnorm = 0.0;
    auto hostAcc = level.f.get_host_access<access::mode::read>();
    for (std::int64_t x = 1; x < level.levelDim[0] + 1; x++) {
        for (std::size_t y = 1; y < level.levelDim[1] + 1; y++) {
            for (std::size_t z = 1; z < level.levelDim[2] + 1; z++) {
                double value = getGpuVal(hostAcc, level.f, x, y, z);
                fnorm += value * value;
            }
        }
    }

    std::cout << "Fnorm: " << fnorm << '\n';

    exit(0);
}

void NewtonSolver::findError(ContextHandles& handles, SyclGridData& grid)
{
    SyclGridData mgGrid = grid; //makeCopyForSolver(handles.queue, grid);
    mgGrid.printProgress = false;

    for (std::size_t i = 1; i < grid.numLevels() - 1; i++) {
        SyclBuffer& src = mgGrid.getLevel(i - 1).newtonV;
        SyclBuffer& dst = mgGrid.getLevel(i).newtonV;
        SyclSolver::restrict(handles.queue, src, dst);
    }

    SyclSolver::solve(handles, mgGrid);

    handles.queue.submit([&](handler& cgh) {
        auto newtonvAcc = grid.getLevel(0).newtonV.get_access<access::mode::read_write>(cgh);
        auto vAcc = mgGrid.getLevel(0).v.get_access<access::mode::read>(cgh);

        cgh.parallel_for<class reset>(range<1>(grid.getLevel(0).newtonV.flatSize()), [newtonvAcc, vAcc](id<1> index) {
            newtonvAcc[index] += vAcc[index];
        });
    });

}

SyclGridData NewtonSolver::makeCopyForSolver(cl::sycl::queue& queue, SyclGridData& grid)
{
    SyclGridData newGrid = grid;
    
    // restrict newtonV to all levels
    // Allocate new buffers for it first
    for (std::size_t i = 0; i < newGrid.numLevels(); i++) {
        SyclGridData::LevelData& level = newGrid.getLevel(i);

        level.newtonV = SyclBuffer(level.newtonV.getXdim(), level.newtonV.getYdim(), level.newtonV.getZdim());
    }
    for (std::size_t i = 1; i < grid.numLevels() - 1; i++) {
        SyclBuffer& src = newGrid.getLevel(i - 1).newtonV;
        SyclBuffer& dst = newGrid.getLevel(i).newtonV;
        SyclSolver::restrict(queue, src, dst);
    }

    return newGrid;
}
