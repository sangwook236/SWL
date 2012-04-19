#if !defined(__SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"
#include <boost/random/linear_congruential.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with univariate normal mixture observation densities

class SWL_RND_UTIL_API HmmWithUnivariateNormalMixtureObservations: public CDHMM, HmmWithMixtureObservations
{
public:
	typedef CDHMM base_type;
	//typedef HmmWithMixtureObservations base_type;

public:
	HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C);
	HmmWithUnivariateNormalMixtureObservations(const size_t K, const size_t C, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const boost::multi_array<double, 2> &alphas, const boost::multi_array<double, 2> &mus, const boost::multi_array<double, 2> &sigmas);
	virtual ~HmmWithUnivariateNormalMixtureObservations();

private:
	HmmWithUnivariateNormalMixtureObservations(const HmmWithUnivariateNormalMixtureObservations &rhs);
	HmmWithUnivariateNormalMixtureObservations & operator=(const HmmWithUnivariateNormalMixtureObservations &rhs);

public:
	//
	boost::multi_array<double, 2> & getMean()  {  return mus_;  }
	const boost::multi_array<double, 2> & getMean() const  {  return mus_;  }
	boost::multi_array<double, 2> & getStandardDeviation()  {  return  sigmas_;  }
	const boost::multi_array<double, 2> & getStandardDeviation() const  {  return  sigmas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::multi_array_ref<double, 2>::array_view<1>::type &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity();
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		HmmWithMixtureObservations::normalizeObservationDensityParameters(K_);
	}

private:
	boost::multi_array<double, 2> mus_;  // the sets of means of each components in the univariate normal mixture distribution
	boost::multi_array<double, 2> sigmas_;  // the sets of standard deviations of each components in the univariate normal mixture distribution

	typedef boost::minstd_rand base_generator_type;
	mutable base_generator_type baseGenerator_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_UNIVARIATE_NORMAL_MIXTURE_OBSERVATIONS__H_
