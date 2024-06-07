#pragma once
#include <vector>
#include <array>

class Vector3 {
public:
	Vector3() = default;
	Vector3(std::size_t x, std::size_t y, std::size_t z);

	void set(std::size_t x, std::size_t y, std::size_t z, double val);
	double get(std::size_t x, std::size_t y, std::size_t z) const;

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

private:
	std::vector<double> values;
	std::array<std::size_t, 3> dims;
};
