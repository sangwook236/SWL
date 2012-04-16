#if !defined(__SWL_POSIX_UTIL__POSIX_TIMER__H_)
#define __SWL_POSIX_UTIL__POSIX_TIMER__H_ 1


#include <sys/time.h>


namespace swl {

//-----------------------------------------------------------------------------
//  PosixTimer

class PosixTimer
{
public:
	PosixTimer();
	~PosixTimer()
	{}

	long long getElapsedTimeInMilliSecond() const;  // milli-second
	long long getElapsedTimeInMicroSecond() const;  // micro-second

private:
	struct timeval startTime_;
};

}  // namespace swl


#endif  // __SWL_POSIX_UTIL__POSIX_TIMER__H_
