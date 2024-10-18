#include "Vector3.h"
#include <assert.h>
#include <iostream>
#include <fstream>

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
	assert(!std::isnan(val) && !std::isinf(val));
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

	for (std::size_t i = 0; i < flatSize(); i++) {
		values[i] += rhs.values[i];
	}
	
	return *this;
}

Vector3& Vector3::operator-=(const Vector3& rhs)
{
	assert(flatSize() == rhs.flatSize());

	for (std::size_t i = 0; i < flatSize(); i++) {
		values[i] -= rhs.values[i];
	}

	return *this;
}

void Vector3::dump(const std::string& file) const
{
	std::ofstream out;
	if (!file.empty()) {
		out.open(file);
	}

	if (out) {
		out << getXdim() << ' ' << getYdim() << ' ' << getZdim() << '\n';
	}

	for (std::size_t x = 0; x < getXdim(); x++) {
		for (std::size_t y = 0; y < getYdim(); y++) {
			for (std::size_t z = 0; z < getZdim(); z++) {
				if (out) {
					out << x << ' ' << y << ' ' << z << ' ' << get(x, y, z) << '\n';
				}else {
					std::cout << x << ' ' << y << ' ' << z << ' ' << get(x, y, z) << '\n';
				}
			}
		}
	}
}
