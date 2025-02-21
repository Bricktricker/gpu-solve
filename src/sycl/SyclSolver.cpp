#include "SyclSolver.h"
#include "../Timer.h"
#include <iostream>
#include <chrono>
#include <string>
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
                }else {
                    std::cout << "Index: " << x << ' ' << y << ' ' << z << " Value: " << getGpuVal(acc, buf, x, y, z) << '\n';
                }
            }
        }
    }
}
}
#endif

void SyclSolver::solve(cl::sycl::queue& queue, SyclGridData& grid)
{
    compResidual(queue, grid, 0);
    double initialResidual = sumBuffer(queue, grid.getLevel(0).r);

    if (grid.printProgress) {
        std::cout << "Inital residual: " << initialResidual << '\n';
    }

    for (std::size_t i = 0; i < grid.maxiter; i++) {
        if (grid.printProgress) {
            Timer::start();
        }

        double res = vcycle(queue, grid);

        if (grid.printProgress) {
            std::cout << "iter: " << i << " residual: " << res << ' ';
            Timer::stop();
        }

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

double SyclSolver::vcycle(queue& queue, SyclGridData& grid)
{
    for (std::size_t i = 0; i < grid.numLevels() - 1; i++) {

        SyclGridData::LevelData& nextLevel = grid.getLevel(i + 1);

        jacobi(queue, grid, i, grid.preSmoothing);

        compResidual(queue, grid, i);

        // restrict residual to next level f
        restrict(queue, grid.getLevel(i).r, nextLevel.f);

        if (grid.mode != GridParams::NONLINEAR) {

            // clear v for next level
            queue.submit([&](handler& cgh) {
                auto vAcc = nextLevel.v.get_access<access::mode::discard_write>(cgh);
                cgh.parallel_for<class reset>(range<1>(nextLevel.v.flatSize()), [=](id<1> index) {
                    vAcc[index] = 0.0;
                });
            });

        }else {
            restrict(queue, grid.getLevel(i).v, nextLevel.restV);
            restrict(queue, grid.getLevel(i).v, nextLevel.v);

            // Compute A^2h (v^2h), and save it in r, so we don't need a new buffer for it
            applyStencil(queue, grid, i + 1, nextLevel.restV);

            // Add A^2h (v^2h) to r^2h
            queue.submit([&](handler& cgh) {
                auto fAcc = nextLevel.f.get_access<access::mode::read_write>(cgh);
                auto rAcc = nextLevel.r.get_access<access::mode::read>(cgh);
                cgh.parallel_for<class sum>(range<1>(nextLevel.f.flatSize()), [=](id<1> index) {
                    fAcc[index] += rAcc[index];
                });
            });
        }
    }

    jacobi(queue, grid, grid.numLevels() - 1, grid.preSmoothing + grid.postSmoothing);

    for (std::size_t i = grid.numLevels() - 1; i > 0; i--) {
        SyclGridData::LevelData& thisLevel = grid.getLevel(i);
        SyclGridData::LevelData& prevLevel = grid.getLevel(i - 1);

        if (grid.mode == GridParams::NONLINEAR) {
            // compute u^2h = u^2h - v^2h
            queue.submit([&](handler& cgh) {
                auto vAcc = thisLevel.v.get_access<access::mode::read_write>(cgh);
                auto restvAcc = thisLevel.restV.get_access<access::mode::read>(cgh);
                cgh.parallel_for<class sum1>(range<1>(thisLevel.v.flatSize()), [=](id<1> index) {
                    vAcc[index] -= restvAcc[index];
                });
            });
        }

        // interpolate v to previous level e
        interpolate(queue, prevLevel.e, thisLevel.v);

        // v = v + e
        queue.submit([&](handler& cgh) {
            auto vAcc = prevLevel.v.get_access<access::mode::read_write>(cgh);
            auto eAcc = prevLevel.e.get_access<access::mode::read>(cgh);
            cgh.parallel_for<class sum2>(range<1>(prevLevel.v.flatSize()), [=](id<1> index) {
                vAcc[index] += eAcc[index];
            });
        });

        jacobi(queue, grid, i - 1, grid.postSmoothing);
    }

    compResidual(queue, grid, 0);
    double res = sumBuffer(queue, grid.getLevel(0).r);
    return res;
}

void SyclSolver::jacobi(queue& queue, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter)
{
    SyclGridData::LevelData& level = grid.getLevel(levelNum);
    const double preFac = grid.stencil.values[0] / (level.h * level.h);
    const double alpha = (level.h * level.h) / grid.stencil.values[0]; // stencil center

    for (std::size_t i = 0; i < maxiter; i++) {
        compResidual(queue, grid, levelNum);

        queue.submit([&](handler& cgh) {
            auto vAcc = level.v.get_access<access::mode::read_write>(cgh);
            auto newtonvAcc = level.newtonV.get_access<access::mode::read>(cgh);
            auto rAcc = level.r.get_access<access::mode::read>(cgh);

            cgh.parallel_for<class jacobiK>(range<1>(level.v.flatSize()), [=, omega=grid.omega, gamma=grid.gamma, mode=grid.mode](id<1> idx) {
                double1 vVal = vAcc[idx[0]];

                double1 newV;
                if (mode == GridParams::LINEAR) {
                    newV = vVal + omega * (alpha * rAcc[idx[0]]);
                }else if(mode == GridParams::NONLINEAR) {
                    double1 ex = cl::sycl::exp(vVal);
                    double1 denuminator = preFac + gamma * (1 + vVal) * ex;

                    newV = vVal + omega * (rAcc[idx[0]] / denuminator);
                }else {
                    // Newton
                    double1 newtonV = newtonvAcc[idx[0]];
                    double1 ex = cl::sycl::exp(newtonV);
                    double1 denuminator = preFac + gamma * (1 + newtonV) * ex;

                    newV = vVal + omega * (rAcc[idx[0]] / denuminator);
                }

                vAcc[idx[0]] = newV;
            });
        });
    }
}

void SyclSolver::compResidual(queue& queue, SyclGridData& grid, std::size_t levelNum)
{
    SyclGridData::LevelData& level = grid.getLevel(levelNum);

    range<3> range(level.levelDim[0], level.levelDim[1], level.levelDim[2]);

    queue.submit([&](handler& cgh) {

        auto fAcc = level.f.get_access<access::mode::read>(cgh);
        auto vAcc = level.v.get_access<access::mode::read>(cgh);
        auto newtonvAcc = level.newtonV.get_access<access::mode::read>(cgh);
        auto rAcc = level.r.get_access<access::mode::write>(cgh);

        cgh.parallel_for<class residual>(range, [=, h=level.h, gamma=grid.gamma, mode=grid.mode, dims=level.v.getDims(), stencil=grid.stencil](id<3> index) {
            double1 stencilsum = 0.0;
            for (std::size_t i = 0; i < stencil.values.size(); i++) {
                const int1 flatIdx = Sycl3dAccesor::flatIndex(dims, index[0] + (stencil.getXOffset(i) + 1), index[1] + (stencil.getYOffset(i) + 1), index[2] + (stencil.getZOffset(i) + 1));
                auto vVal = vAcc[flatIdx];
                stencilsum += stencil.values[i] * vVal;
            }

            int1 centerIdx = Sycl3dAccesor::shift1Index(dims, index);
            stencilsum /= h * h;

            if (mode == GridParams::NEWTON) {
                double1 newtonV = newtonvAcc[centerIdx];
                double1 ex = cl::sycl::exp(newtonV);
                double1 nonLinear = gamma * (1 + newtonV) * vAcc[centerIdx] * ex;
                stencilsum += nonLinear;
            }
            else if (mode == GridParams::NONLINEAR) {
                // See tutorial_multigrid.pdf, page 102, Formula 6.13
                double1 vVal = vAcc[centerIdx];
                double1 ex = cl::sycl::exp(vVal);
                double1 nonLinear = gamma * vVal * ex;
                stencilsum += nonLinear;
            }

            rAcc[centerIdx] = fAcc[centerIdx] - stencilsum;
        });
    });
}

// save result in 'r'. Only needed for non-linear
void SyclSolver::applyStencil(cl::sycl::queue& queue, SyclGridData& grid, std::size_t levelNum, SyclBuffer& v)
{
    assert(grid.mode == GridParams::NONLINEAR);
    SyclGridData::LevelData& level = grid.getLevel(levelNum);
    assert(level.v.flatSize() == v.flatSize());
    SyclBuffer& result = level.r;

    range<3> range(level.levelDim[0], level.levelDim[1], level.levelDim[2]);

    queue.submit([&](handler& cgh) {

        auto vAcc = v.get_access<access::mode::read>(cgh);
        auto resultAcc = result.get_access<access::mode::write>(cgh);

        cgh.parallel_for<class apply>(range, [=, h=level.h, dims=v.getDims(), stencil=grid.stencil, gamma=grid.gamma](id<3> index) {
            
            double1 stencilsum = 0.0;
            for (std::size_t i = 0; i < stencil.values.size(); i++) {
                const int1 flatIdx = Sycl3dAccesor::flatIndex(dims, index[0] + (stencil.getXOffset(i) + 1), index[1] + (stencil.getYOffset(i) + 1), index[2] + (stencil.getZOffset(i) + 1));
                auto vVal = vAcc[flatIdx];
                stencilsum += stencil.values[i] * vVal;
            }
            stencilsum /= h * h;

            int1 centerIdx = Sycl3dAccesor::shift1Index(dims, index);

            // See tutorial_multigrid.pdf, page 102, Formula 6.13
            double1 vVal = vAcc[centerIdx];
            double1 ex = cl::sycl::exp(vVal);
            double1 nonLinear = gamma * vVal * ex;
            stencilsum += nonLinear;

            resultAcc[centerIdx] = stencilsum;
        });
    });
}

double SyclSolver::sumBuffer(queue& queue, SyclBuffer& buffer)
{
    // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/reduction.html
    Timer::push("sumBuffer");

    std::size_t flatSize = buffer.flatSize();

    std::size_t num_work_items = 1;
    bool skipFirst; // If the flat size is odd, we skip the first value in the buffer during the sum reduction, and copy it later into the accumulation buffer
    if (flatSize % 2 != 0) {
        assert(buffer.flatSize() % 2 != 0); // flat size is odd
        // Ignore the first element, so we get an even number of items in the buffer
        flatSize--;

        while (true) {
            std::size_t next = num_work_items * 2;
            if (flatSize % next != 0) {
                break;
            }
            if (next > 512) {
                break;
            }
            num_work_items = next;
        }

        skipFirst = true;
    }
    else {
        // Not all dimensions are odd
        assert(false); // TODO: Implement and set num_work_items
        skipFirst = false;
    }


    cl::sycl::buffer<double> accumBuf(num_work_items);

    queue.submit([&](handler& cgh) {
        auto accumAcc = accumBuf.get_access<access::mode::discard_write>(cgh);
        auto accR = buffer.get_access<access::mode::read>(cgh);

        cgh.parallel_for<class sumK>(range<1>(num_work_items), [=](id<1> index) {
            double1 sum = 0;
            SYCL_FOR(int1 i = index[0], i < flatSize, i) { // can't used i += num_work_items here, breaks kernel generation
                // Don't use SYCL_IF, we can decide that while building the kernel
                if (skipFirst) {
                    double1 val = accR[i + 1];
                    val = val * val;
                    sum += val;
                }
                else {
                    double1 val = accR[i];
                    val = val * val;
                    sum += val;
                }

                i += num_work_items;
            }
            SYCL_END;

            accumAcc[index[0]] = sum;
        });
    });

    if (skipFirst) {
        queue.submit([&](handler& cgh) {
            auto accumAcc = accumBuf.get_access<access::mode::read_write>(cgh);
            auto accR = buffer.get_access<access::mode::read>(cgh);

            cgh.single_task<class first>([=]() {
                double1 val = accR[0];
                val = val * val;
                accumAcc[0] += val;
            });
        });
    }

#ifdef SYCL_GTX
    auto accumAcc = accumBuf.get_access<access::mode::read, access::target::host_buffer>();
#else
    sycl::host_accessor accumAcc{ accumBuf, sycl::read_only };
#endif


    double sum = 0;
    for (int i = 0; i < num_work_items; i++) {
        sum += accumAcc[i];
    }

    Timer::pop("sumBuffer");
    return ::sqrt(sum);
}

void SyclSolver::restrict(queue& queue, SyclBuffer& fine, SyclBuffer& coarse)
{
    queue.submit([&](handler& cgh) {
        auto fineAcc = fine.get_access<access::mode::read>(cgh);
        auto coraseAcc = coarse.get_access<access::mode::write>(cgh);

        range<3> range(coarse.getXdim() - 2, coarse.getYdim() - 2, coarse.getZdim() - 2);

        cgh.parallel_for<class rest>(range, [=, fineDims = fine.getDims(), coarseDims = coarse.getDims()](id<3> index) {
            int1 xCenter = 2 * (index[0] + 1);
            int1 yCenter = 2 * (index[1] + 1);
            int1 zCenter = 2 * (index[2] + 1);

            double1 coarseValue = 0.0;

            for (int ii = -2 + 1; ii < 2; ii++) {
                for (int jj = -2 + 1; jj < 2; jj++) {
                    for (int kk = -2 + 1; kk < 2; kk++) {
                        double fac = 0.125 * ((2.0 - abs(ii)) / 2.0) * ((2.0 - abs(jj)) / 2.0) * ((2.0 - abs(kk)) / 2.0);
                        double1 fineVal = fineAcc[Sycl3dAccesor::flatIndex(fineDims, xCenter + ii, yCenter + jj, zCenter + kk)];
                        coarseValue += fac * fineVal;
                    }
                }
            }

            int1 centerIdxCoarse = Sycl3dAccesor::shift1Index(coarseDims, index);
            coraseAcc[centerIdxCoarse] = coarseValue;
        });
    });

}

void SyclSolver::interpolate(queue& queue, SyclBuffer& fine, SyclBuffer& coarse)
{
    // prepare
    queue.submit([&](handler& cgh) {
        auto coarseAcc = coarse.get_access<access::mode::read>(cgh);
        auto fineAcc = fine.get_access<access::mode::write>(cgh);

        range<3> rangePrep(fine.getXdim() / 2, fine.getYdim() / 2, fine.getZdim() / 2);
        cgh.parallel_for<class prep>(rangePrep, [=, fineDims=fine.getDims(), coarseDims=coarse.getDims()](id<3> index) {
            int1 x = index[0] * 2;
            int1 y = index[1] * 2;
            int1 z = index[2] * 2;
            fineAcc[Sycl3dAccesor::flatIndex(fineDims, x, y, z)] = coarseAcc[Sycl3dAccesor::flatIndex(coarseDims, index)];
        });
    });

    // Interpolate in x-direction
    queue.submit([&](handler& cgh) {
        auto fineAcc = fine.get_access<access::mode::read_write>(cgh);

        range<3> rangeX(fine.getXdim() / 2, fine.getYdim() / 2 + 1, fine.getZdim() / 2 + 1);
        cgh.parallel_for<class InteX>(rangeX, [=, dims=fine.getDims()](id<3> index) {
            int1 x = index[0] * 2;
            int1 y = index[1] * 2;
            int1 z = index[2] * 2;
            double1 val = 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x, y, z)] + 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x + 2, y, z)];
            fineAcc[Sycl3dAccesor::flatIndex(dims, x + 1, y, z)] = val;
        });
    });

    // Interpolate in y-direction
    queue.submit([&](handler& cgh) {
        auto fineAcc = fine.get_access<access::mode::read_write>(cgh);

        range<3> rangeY(fine.getXdim(), fine.getYdim() / 2, fine.getZdim() / 2 + 1);
        cgh.parallel_for<class InteY>(rangeY, [=, dims = fine.getDims()](id<3> index) {
            int1 x = index[0];
            int1 y = index[1] * 2;
            int1 z = index[2] * 2;
            double1 val = 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x, y, z)] + 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x, y + 2, z)];
            fineAcc[Sycl3dAccesor::flatIndex(dims, x, y + 1, z)] = val;
        });
    });

    // Interpolate in z-direction
    queue.submit([&](handler& cgh) {
        auto fineAcc = fine.get_access<access::mode::read_write>(cgh);

        range<3> rangeZ(fine.getXdim(), fine.getYdim(), fine.getZdim() / 2);
        cgh.parallel_for<class InteZ>(rangeZ, [=, dims = fine.getDims()](id<3> index) {
            int1 x = index[0];
            int1 y = index[1];
            int1 z = index[2] * 2;
            double1 val = 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x, y, z)] + 0.5 * fineAcc[Sycl3dAccesor::flatIndex(dims, x, y, z + 2)];
            fineAcc[Sycl3dAccesor::flatIndex(dims, x, y, z + 1)] = val;
        });
    });
}
