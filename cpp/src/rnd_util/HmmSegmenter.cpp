#include "swl/Config.h"
#include "swl/rnd_util/HmmSegmenter.h"
#include "swl/rnd_util/DDHMM.h"
#include "swl/rnd_util/CDHMM.h"
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// HMM Segmenter

/*static*/ bool HmmSegmenter::segmentByViterbiAlgorithm(const boost::scoped_ptr<DDHMM> &ddhmm, const uivector_type &observations, double &logProbability, std::size_t &startStateIndex, std::size_t &endStateIndex, uivector_type &states)
{
	throw std::runtime_error("not yet implemented");

	return false;
}

/*static*/ bool HmmSegmenter::segmentByViterbiAlgorithm(const boost::scoped_ptr<CDHMM> &cdhmm, const dmatrix_type &observations, double &logProbability, std::size_t &startStateIndex, std::size_t &endStateIndex, uivector_type &states)
{
	const size_t K = cdhmm->getStateDim();  // the number of hidden states
	const size_t D = cdhmm->getObservationDim();  // the number of observation symbols
	//const size_t D = observations.size2();  // the number of observation symbols
	const size_t N = observations.size1();  // the length of observation sequence

	swl::CDHMM::dmatrix_type delta(N, K, 0.0);
	swl::CDHMM::uimatrix_type psi(N, K, (unsigned int)-1);
	cdhmm->runViterbiAlgorithm(N, observations, delta, psi, states, logProbability, true);

	throw std::runtime_error("not yet implemented");

	return false;
}

}  // namespace swl
