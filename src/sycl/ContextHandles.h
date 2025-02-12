#pragma once
#include <CL/sycl.hpp>
#include <iostream>

struct ContextHandles {

	static ContextHandles init() {

        auto platforms = cl::sycl::platform::get_platforms();
        std::cout << "Number of platforms: " << platforms.size() << '\n';
        std::size_t platformIdx = 0;
        std::size_t deviceIdx = 0;
        bool foundDevice = false;
        for (std::size_t i = 0; i < platforms.size(); i++) {
            const cl::sycl::platform& P = platforms[i];

            const auto devices = P.get_devices(cl::sycl::info::device_type::all);
            for (std::size_t j = 0; j < devices.size(); j++) {
                const auto& device = devices[j];
                if (device.is_gpu()) {
                    platformIdx = i;
                    deviceIdx = j;
                    foundDevice = true;
                    break;
                }
            }
            if (foundDevice) break;
        }
        const cl::sycl::platform& P = platforms.at(platformIdx);
        auto platformName = P.get_info<cl::sycl::info::platform::name>();
        std::cout << "Platform: " << platformName << '\n';

        const auto devices = P.get_devices(cl::sycl::info::device_type::all);
        const cl::sycl::device& D = devices.at(deviceIdx);
        std::cout << "Device: " << D.get_info<cl::sycl::info::device::name>() << '\n';

        return ContextHandles(D);
	}

    ContextHandles(const cl::sycl::device& _device)
        : device(_device), context(device), queue(context, device)
    {}

	cl::sycl::device device;
	cl::sycl::context context;
	cl::sycl::queue queue;
};
