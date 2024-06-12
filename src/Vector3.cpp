#include "Vector3.h"
#include <assert.h>

Vector3::Vector3(std::size_t x, std::size_t y, std::size_t z)
{
	dims[0] = x;
	dims[1] = y;
	dims[2] = z;
	values.resize(x * y * z);
}

void Vector3::set(std::size_t x, std::size_t y, std::size_t z, double val)
{
	const std::size_t idx = z + y * dims[2] + x * dims[2] * dims[1];
	assert(idx < values.size());
	values[idx] = val;
}

double Vector3::get(std::size_t x, std::size_t y, std::size_t z) const
{
	const std::size_t idx = z + y * dims[2] + x * dims[2] * dims[1];
	assert(idx < values.size());
	return values[idx];
}

void Vector3::fill(double val)
{
	std::fill(values.begin(), values.end(), val);
}

Vector3& Vector3::operator+=(const Vector3& rhs)
{
	assert(flatSize() == rhs.flatSize());

	for (size_t i = 0; i < flatSize(); i++) {
		values[i] += rhs.values[i];
	}
	
	return *this;
}
