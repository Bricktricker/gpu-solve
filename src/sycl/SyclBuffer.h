#pragma once
#include <CL/sycl.hpp>
#include <array>

class SyclBuffer; // Forward declaration

template<cl::sycl::access::mode mode, cl::sycl::access::target target>
class Sycl3dAccesor
{
	friend class SyclBuffer;
protected:
	Sycl3dAccesor(cl::sycl::accessor<float, 1, mode, target>& _acc, const std::array<std::size_t, 3>& _dims)
		: acc(_acc), dims(_dims)
	{}

public:
	cl::sycl::detail::data_ref operator()(cl::sycl::id<3>& idx3) const
	{
		const cl::sycl::int1 idx1 = idx3[2] * (dims[1] * dims[0]) + idx3[1] * dims[0] + idx3[0];
		return acc[idx1];
	}

	template<class point_ref_x, class point_ref_y, class point_ref_z>
	cl::sycl::detail::data_ref operator()(point_ref_x x, point_ref_y y, point_ref_z z) const
	{
		const cl::sycl::int1 idx1 = z * (dims[1] * dims[0]) + y * dims[0] + x;
		return acc[idx1];
	}

	cl::sycl::int1 getIndex(cl::sycl::id<3>& idx3) const
	{
		return idx3[2] * (dims[1] * dims[0]) + idx3[1] * dims[0] + idx3[0];
	}

	cl::sycl::detail::data_ref shift1(cl::sycl::id<3>& idx3) const
	{
		const cl::sycl::int1 idx1 = (idx3[2]+1) * (dims[1] * dims[0]) + (idx3[1]+1) * dims[0] + (idx3[0]+1);
		return acc[idx1];
	}

	template<class point_ref_x, class point_ref_y, class point_ref_z>
	cl::sycl::detail::data_ref shift1(point_ref_x x, point_ref_y y, point_ref_z z) const
	{
		const cl::sycl::int1 idx1 = (z + 1) * (dims[1] * dims[0]) + (y + 1) * dims[0] + (x + 1);
		return acc[idx1];
	}

	cl::sycl::int1 shift1Index(cl::sycl::id<3>& idx3) const
	{
		return (idx3[2] + 1) * (dims[1] * dims[0]) + (idx3[1] + 1) * dims[0] + (idx3[0] + 1);
	}

	template<class point_ref_x, class point_ref_y, class point_ref_z>
	cl::sycl::int1 shift1Index(point_ref_x x, point_ref_y y, point_ref_z z) const
	{
		return (z + 1) * (dims[1] * dims[0]) + (y + 1) * dims[0] + (x + 1);
	}

	cl::sycl::detail::data_ref operator[](cl::sycl::int1& idx) const
	{
		return acc[idx];
	}

	cl::sycl::detail::data_ref operator[](cl::sycl::detail::data_ref& idx) const
	{
		return acc[idx];
	}

private:
	cl::sycl::accessor<float, 1, mode, target> acc;
	std::array<std::size_t, 3> dims;
};

class SyclBuffer
{
public:
	SyclBuffer() = default;
	SyclBuffer(std::size_t x, std::size_t y, std::size_t z)
		: buffer(cl::sycl::range<1>(x* y* z))
	{
		dims[0] = x;
		dims[1] = y;
		dims[2] = z;
	}

	template<cl::sycl::access::mode mode, cl::sycl::access::target target = cl::sycl::access::target::global_buffer>
	Sycl3dAccesor<mode, target> get_access(cl::sycl::handler& cgh)
	{
		return Sycl3dAccesor{ buffer.get_access<mode, target>(cgh), dims };
	}

	template<cl::sycl::access::mode mode>
	cl::sycl::accessor<float, 1, mode, cl::sycl::access::target::host_buffer> get_host_access()
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

	cl::sycl::buffer<float, 1>& handle()
	{
		return buffer;
	}

	cl::sycl::range<3> getRange() const
	{
		return cl::sycl::range<3>(dims[0], dims[1], dims[2]);
	}

private:
	cl::sycl::buffer<float, 1> buffer;
	std::array<std::size_t, 3> dims;
};