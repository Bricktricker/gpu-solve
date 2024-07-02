#pragma once
#include <string>
#include <chrono>
#include <unordered_map>

class Timer
{
	struct PartialTime {
		uint64_t timeMs;
		std::chrono::high_resolution_clock::time_point lastStart;
	};

public:
	static void start();
	static void stop();

	static void push(const std::string& name);
	static void pop(const std::string& name);

private:
	static std::chrono::high_resolution_clock::time_point startPoint;
	static std::unordered_map<std::string, PartialTime> times;
};