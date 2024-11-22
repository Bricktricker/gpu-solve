#pragma once
#include <CL/sycl.hpp>
#include <array>
#include "sycl_compat.h"

class SyclBuffer; // Forward declaration
using BufferDim = std::array<std::size_t, 3>;

class Sycl3dAccesor
{
	friend class SyclBuffer;

public:
	Sycl3dAccesor() = delete;

	static int1 flatIndex(const BufferDim& dims, cl::sycl::id<3>& idx3)
	{
		return idx3[2] * (dims[1] * dims[0]) + idx3[1] * dims[0] + idx3[0];
	}

#ifdef SYCL_GTX_TARGET
	template<class point_ref_x, class point_ref_y, class point_ref_z>
	static int1 flatIndex(const BufferDim& dims, const point_ref_x& x, const point_ref_y& y, const point_ref_z& z)
#else
	static int flatIndex(const BufferDim& dims, const int x, const int y, const int z)
#endif
	{
		return z * (dims[1] * dims[0]) + y * dims[0] + x;
	}

	static int1 shift1Index(const BufferDim& dims, cl::sycl::id<3>& idx3)
	{
		return (idx3[2]+1) * (dims[1] * dims[0]) + (idx3[1]+1) * dims[0] + (idx3[0]+1);
	}

#ifdef SYCL_GTX_TARGET
	template<class point_ref_x, class point_ref_y, class point_ref_z>
	static cl::sycl::detail::data_ref shift1Index(const BufferDim& dims, const point_ref_x& x, const point_ref_y& y, const point_ref_z& z)
#else
	static int shift1Index(const BufferDim& dims, const int x, const int y, const int z)
#endif
	{
		return (z + 1) * (dims[1] * dims[0]) + (y + 1) * dims[0] + (x + 1);
	}

};

class SyclBuffer
{
public:

	SyclBuffer(std::size_t x, std::size_t y, std::size_t z)
		: buffer(cl::sycl::range<1>(x* y* z))
	{
		dims[0] = x;
		dims[1] = y;
		dims[2] = z;
	}

	template<cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
	cl::sycl::accessor<double, 1, mode, target> get_access(cl::sycl::handler& cgh)
	{
		return buffer.get_access<mode, target>(cgh);
	}

	template<cl::sycl::access::mode mode>
	cl::sycl::accessor<double, 1, mode, cl::sycl::access::target::host_buffer> get_host_access()
	{
		return buffer.get_access<mode, cl::sycl::access::target::host_buffer>();
	}

	std::size_t getXdim() const
	{
		return dims[0];
	}
	std::size_t getYdim() const
	{
		return dims[1];
	}
	std::size_t getZdim() const
	{
		return dims[2];
	}
	std::size_t flatSize() const
	{
		return dims[0] * dims[1] * dims[2];
	}

	cl::sycl::range<3> getRange() const
	{
		return cl::sycl::range<3>(dims[0], dims[1], dims[2]);
	}

	const BufferDim& getDims() const
	{
		return dims;
	}

private:
	cl::sycl::buffer<double, 1> buffer;
	BufferDim dims;
};