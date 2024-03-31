#include "common.h"
#include <chrono>

using stdclock = std::chrono::steady_clock;

#ifdef __MACH__
#include <mach/mach_time.h>
inline u64 cpu_timer()
{
	return mach_absolute_time();
}

inline u64 ticks_to_ns(u64 time)
{
	static mach_timebase_info_data_t timebase_info;
    if (timebase_info.denom == 0) 
        mach_timebase_info(&timebase_info);
    return time* timebase_info .numer / timebase_info.denom;
}
#endif
#ifdef __linux__
#include <x86intrin.h>
inline u64 cpu_timer()
{
	return __rdtsc();
}
#endif
#ifdef _WIN32
#include <intrin.h>
inline u64 cpu_timer()
{
	return __rdtsc();
}
#endif


constexpr u64 wait_time{100};

// ticks per seconds
inline u64 cpu_frequency()
{
	static u64 freq{0};
	if(!freq){
		u64 s = cpu_timer();
		auto start = stdclock::now();
		auto end   = start + std::chrono::milliseconds(wait_time);
		while(stdclock::now() < end) {;}
	 	u64 e = cpu_timer();	
		freq = (e-s)* (u64)(1000. / wait_time); 
	}	
	return freq;
}

void print_time(u64 time)
{
	printf("\tTime taken: %llu cycles in %f ms\n", time, (f64)time/cpu_frequency()*1.e-3);
}

