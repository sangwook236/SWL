#if !defined(__SWL_WIN_UTIL__WIN_TIMER__H_)
#define __SWL_WIN_UTIL__WIN_TIMER__H_ 1


#include <windows.h>

namespace swl {

//-----------------------------------------------------------------------------
//  WinTimer

class WinTimer
{
public:
	WinTimer()
	{
		freq_.LowPart = 0;
		freq_.HighPart = 0;
		startTime_.LowPart = 0;
		startTime_.HighPart = 0;
		QueryPerformanceFrequency(&freq_);
		QueryPerformanceCounter(&startTime_);
	}
	~WinTimer()
	{
	}

	double getElapsedTimeInMilliSecond() const  // milli-second
	{
		if (0 == freq_.HighPart && 0 == freq_.LowPart) return 0.0;

		LARGE_INTEGER endTime;
		endTime.LowPart = 0;
		endTime.HighPart = 0;
		QueryPerformanceCounter(&endTime);
		return (0 == endTime.HighPart && 0 == endTime.LowPart) ? 0.0 : ((endTime.QuadPart - startTime_.QuadPart) * 1000.0 / freq_.QuadPart);
	}

	double getElapsedTimeInMicroSecond() const  // micro-second
	{
		if (0 == freq_.HighPart && 0 == freq_.LowPart) return 0.0;

		LARGE_INTEGER endTime;
		endTime.LowPart = 0;
		endTime.HighPart = 0;
		QueryPerformanceCounter(&endTime);
		return (0 == endTime.HighPart && 0 == endTime.LowPart) ? 0.0 : ((endTime.QuadPart - startTime_.QuadPart) * 1000000.0 / freq_.QuadPart);
	}

private:
	LARGE_INTEGER freq_;
	LARGE_INTEGER startTime_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__WIN_TIMER__H_
