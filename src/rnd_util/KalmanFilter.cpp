#include "swl/Config.h"
#include "swl/rnd_util/KalmanFilter.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

KalmanFilter::KalmanFilter()
{
}

KalmanFilter::~KalmanFilter()
{
}

}  // namespace swl
