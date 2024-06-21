#include "SyclSolver.h"
#include <iostream>

using namespace cl::sycl;

void SyclSolver::solve(SyclGridData& grid)
{
    auto platforms = platform::get_platforms();
    std::cout << "Number of platforms: " << platforms.size() << '\n';
    platform P = platforms.at(0);

    const auto devices = P.get_devices(info::device_type::gpu);
    device D = devices[0];
    
    {
        queue queue(D);
        
        // TODO: move everything into one submit?
        queue.submit([&](handler& cgh) {
            grid.initBuffers(cgh);
        });

        queue.submit([&](handler& cgh) {
            compResidual(cgh, grid, 0);
        });

        float res = sumResidual(queue, grid, 0);
        std::cout << "Inital residual: " << res << '\n';
    }
    
}

void SyclSolver::vsycle(handler& cgh, SyclGridData& grid)
{
    for (std::size_t i = 0; i < grid.numLevels() - 1; i++) {
        
    }
}

void SyclSolver::jacobi(handler& cgh, SyclGridData& grid, std::size_t levelNum, std::size_t maxiter)
{

}

void SyclSolver::compResidual(handler& cgh, SyclGridData& grid, std::size_t levelNum)
{
    SyclGridData::LevelData& level = grid.getLevel(levelNum);

    range<3> range(level.levelDim[0], level.levelDim[1], level.levelDim[2]);

    auto fAcc = level.f.get_access<access::mode::read>(cgh);
    auto vAcc = level.v.get_access<access::mode::read>(cgh);
    auto rAcc = level.r.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class res>(range, [=](id<3> index) {
        int1 centerIdx = vAcc.shift1Index(index);

        float1 stencilsum = 6.0f * vAcc[centerIdx];
        stencilsum += -1.f * vAcc.shift1(index[0] - 1, index[1], index[2]); // left
        stencilsum += -1.f * vAcc.shift1(index[0] + 1, index[1], index[2]); // right
        stencilsum += -1.f * vAcc.shift1(index[0], index[1] - 1, index[2]); // bottom
        stencilsum += -1.f * vAcc.shift1(index[0], index[1] + 1, index[2]); // top
        stencilsum += -1.f * vAcc.shift1(index[0], index[1], index[2] - 1); // front
        stencilsum += -1.f * vAcc.shift1(index[0], index[1], index[2] + 1); // back

        rAcc[centerIdx] = fAcc[centerIdx] - stencilsum;
    });
}

float SyclSolver::sumResidual(queue& queue, SyclGridData& grid, std::size_t levelNum)
{
    // https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/reduction.html
    
    SyclGridData::LevelData& level = grid.getLevel(levelNum);

    std::size_t flatSize = level.r.flatSize();

    std::size_t num_work_items;
    bool skipFirst; // If the flat size is odd, we skip the first value in the buffer during the sum reduction, and copy it later into the accumulation buffer
    if (flatSize % 2 != 0) {
        assert(level.r.flatSize() % 2 != 0); // flat size is odd
        // Ignore the first element, so we get an even number of items in the buffer
        flatSize--;

        num_work_items = 2;
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


    buffer<float> accumBuf(num_work_items);

    queue.submit([&](handler& cgh) {
        // write needed for initalization
        auto accR = level.r.handle().get_access<access::mode::read_write>(cgh);

        // square residual first
        cgh.parallel_for<class sqare>(range<1>(level.r.flatSize()), [=](id<1> index) {
            cl::sycl::float1 val = accR[index];
            accR[index] = val * val;
        });

        auto accumAcc = accumBuf.get_access<access::mode::discard_write>(cgh);

        cgh.parallel_for<class sum>(range<1>(num_work_items), [=](id<1> index) {
            float1 sum = 0;
            SYCL_FOR(int1 i = index[0], i < flatSize, i) { // can't used i += num_work_items here, breaks kernel generation
                // Don't use SYCL_IF, we can decide that while building the kernel
                if (skipFirst) {
                    sum += accR[i+1];
                }else {
                    sum += accR[i];
                }

                i += num_work_items;
            }
            SYCL_END;

            accumAcc[index[0]] = sum;
        });

        if (skipFirst) {
            cgh.single_task<class first>([=]() {
                accumAcc[0] += accR[0];
            });
        }
    });

    auto accumAcc = accumBuf.get_access<access::mode::read, access::target::host_buffer>();
    float sum = 0;
    for (int i = 0; i < num_work_items; i++) {
        sum += accumAcc[i];
    }

    return sqrt(sum);
}
