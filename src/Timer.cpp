#include "Timer.h"
#include <iostream>

std::chrono::high_resolution_clock::time_point Timer::startPoint{};
std::unordered_map<std::string, Timer::PartialTime> Timer::times;

void Timer::start()
{
	times.clear();
	startPoint = std::chrono::high_resolution_clock::now();
}

void Timer::stop()
{
	auto end = std::chrono::high_resolution_clock::now();
	const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - startPoint).count();
	std::cout << "Took " << time << "ms, partials: ";

	for (const auto& [name, t] : times) {
		std::cout << name << ": " << t.timeMs << "ms ";
	}

	std::cout << '\n';
}

void Timer::push(const std::string& name)
{
	Timer::PartialTime& t = times[name];
	t.lastStart = std::chrono::high_resolution_clock::now();
}

void Timer::pop(const std::string& name)
{
	auto end = std::chrono::high_resolution_clock::now();
	Timer::PartialTime& t = times.at(name);
	const auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - t.lastStart).count();
	t.timeMs += time;
}
