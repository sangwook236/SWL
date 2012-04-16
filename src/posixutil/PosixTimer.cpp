#include "swl/posixutil/PosixTimer.h"


namespace swl {

//-----------------------------------------------------------------------------
//  PosixTimer

PosixTimer::PosixTimer()
{
    gettimeofday(&startTime_, 0L);
}

long long PosixTimer::getElapsedTimeInMilliSecond() const  // milli-second
{
    struct timeval endTime;
    gettimeofday(&endTime, 0L);

    const long long sec = endTime.tv_sec - startTime_.tv_sec;
    const long long usec = endTime.tv_usec - startTime_.tv_usec;

    return (long long)(((sec) * 1000 + usec / 1000.0) + 0.5);
}

long long PosixTimer::getElapsedTimeInMicroSecond() const  // micro-second
{
    struct timeval endTime;
    gettimeofday(&endTime, 0L);

    long long sec = endTime.tv_sec - startTime_.tv_sec;
    long long usec = endTime.tv_usec - startTime_.tv_usec;

    return (long long)((sec) * 1000000 + usec);
}

}  // namespace swl
