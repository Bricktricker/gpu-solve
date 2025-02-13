#pragma once
#include <CL/sycl.hpp>
#include <iostream>

struct ContextHandles {

	static ContextHandles init() {

        auto platforms = cl::sycl::platform::get_platforms();
        std::cout << "Number of platforms: " << platforms.size() << '\n';
        std::size_t platformIdx = 0;
        std::size_t deviceIdx = 0;
        for (std::size_t i = 0; i < platforms.size(); i++) {
            const cl::sycl::platform& P = platforms[i];
            std::cout << "\t" << (i+1) << ". Platform: " << P.get_info<cl::sycl::info::platform::name>() << '\n';

            const auto devices = P.get_devices(cl::sycl::info::device_type::all);
            for (std::size_t j = 0; j < devices.size(); j++) {
                const auto& device = devices[j];
                std::cout << "\t\t" << (j+1) << ". Device: " << device.get_info<cl::sycl::info::device::name>() << '\n';
                if (device.is_gpu()) {
                    platformIdx = i;
                    deviceIdx = j;
                }
            }
        }
        const cl::sycl::platform& P = platforms.at(platformIdx);
        auto platformName = P.get_info<cl::sycl::info::platform::name>();
        std::cout << "Selected " << (platformIdx+1) << ". Platform: " << platformName << '\n';

        const auto devices = P.get_devices(cl::sycl::info::device_type::all);
        const cl::sycl::device& D = devices.at(deviceIdx);
        std::cout << "Selected " << (deviceIdx+1) << ". Device: " << D.get_info<cl::sycl::info::device::name>() << '\n';

        return ContextHandles(D);
	}

    ContextHandles(const cl::sycl::device& _device)
        : device(_device), context(device), queue(context, device)
    {}

	cl::sycl::device device;
	cl::sycl::context context;
	cl::sycl::queue queue;
};
