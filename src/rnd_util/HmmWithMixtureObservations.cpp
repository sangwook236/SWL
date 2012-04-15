#include "swl/Config.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithMixtureObservations::HmmWithMixtureObservations(const size_t C)
: C_(C), alpha_(C, 0.0)  // 0-based index
//: C_(C), alpha_(boost::extents[boost::multi_array_types::extent_range(1, C+1)])  // 1-based index
{
}

HmmWithMixtureObservations::HmmWithMixtureObservations(const size_t C, const std::vector<double> &alpha)
: C_(C), alpha_(alpha)
{
}

HmmWithMixtureObservations::~HmmWithMixtureObservations()
{
}

}  // namespace swl
