#include "swl/Config.h"
#include "swl/rnd_util/HmmWithVonMisesObservations.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K)
: base_type(K, 1), mus_(K, 0.0), kappas_(K, 0.0)  // 0-based index
{
}

HmmWithVonMisesObservations::HmmWithVonMisesObservations(const size_t K, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &mus, const std::vector<double> &kappas)
: base_type(K, 1, pi, A), mus_(mus), kappas_(kappas)
{
}

HmmWithVonMisesObservations::~HmmWithVonMisesObservations()
{
}

bool HmmWithVonMisesObservations::estimateParameters(const size_t N, const boost::multi_array<double, 2> &observations, const double terminationTolerance, boost::multi_array<double, 2> &alpha, boost::multi_array<double, 2> &beta, boost::multi_array<double, 2> &gamma, size_t &numIteration, double &initLogProbability, double &finalLogProbability)
{
	throw std::runtime_error("not yet implemented");
}

double HmmWithVonMisesObservations::evaluateEmissionProbability(const int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const
{
	throw std::runtime_error("not yet implemented");
}

void HmmWithVonMisesObservations::generateObservationsSymbol(const int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const bool setSeed /*= false*/) const
{
	throw std::runtime_error("not yet implemented");
}

bool HmmWithVonMisesObservations::readObservationDensity(std::istream &stream)
{
	std::runtime_error("not yet implemented");
	return false;
}

bool HmmWithVonMisesObservations::writeObservationDensity(std::ostream &stream) const
{
	std::runtime_error("not yet implemented");
	return false;
}

void HmmWithVonMisesObservations::initializeObservationDensity()
{
	std::runtime_error("not yet implemented");
}

}  // namespace swl
