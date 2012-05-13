//#include "stdafx.h"
#include "swl/Config.h"
#include "swl/rnd_util/HmmSegmenter.h"
#include "swl/rnd_util/DDHMM.h"
#include "swl/rnd_util/CDHMM.h"
#include <iostream>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

void ddhmm_segmentation_by_viterbi_algorithm()
{
	boost::scoped_ptr<swl::DDHMM> cdhmm;

	// load a model
	{
	}

	//
	swl::HmmSegmenter::uivector_type observations;

	// load an observation sequence
	{
	}

	//
	const size_t N = observations.size();  // the length of observation sequence

	double logProbability = -std::numeric_limits<double>::max();
	std::size_t startStateIndex = (std::size_t)-1, endStateIndex = (std::size_t)-1;
	swl::HmmSegmenter::uivector_type states(N, (unsigned int)-1);

	const bool retval = swl::HmmSegmenter::segmentByViterbiAlgorithm(cdhmm, observations, logProbability, startStateIndex, endStateIndex, states);

	throw std::runtime_error("not yet implemented");
}

void cdhmm_segmentation_by_viterbi_algorithm()
{
	boost::scoped_ptr<swl::CDHMM> cdhmm;

	// load a model
	{
	}

	//
	swl::HmmSegmenter::dmatrix_type observations;

	// load an observation sequence
	{
	}

	//
	const size_t N = observations.size1();  // the length of observation sequence

	double logProbability = -std::numeric_limits<double>::max();
	std::size_t startStateIndex = (std::size_t)-1, endStateIndex = (std::size_t)-1;
	swl::HmmSegmenter::uivector_type states(N, (unsigned int)-1);

	const bool retval = swl::HmmSegmenter::segmentByViterbiAlgorithm(cdhmm, observations, logProbability, startStateIndex, endStateIndex, states);
}

}  // namespace local
}  // unnamed namespace

void hmm_segmentation()
{
	std::cout << "===== HMM Segmentation =====" << std::endl;

	local::ddhmm_segmentation_by_viterbi_algorithm();
	local::cdhmm_segmentation_by_viterbi_algorithm();
}
