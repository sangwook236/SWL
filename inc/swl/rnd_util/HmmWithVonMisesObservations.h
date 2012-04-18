#if !defined(__SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with von Mises observation densities

class SWL_RND_UTIL_API HmmWithVonMisesObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

public:
	HmmWithVonMisesObservations(const size_t K);
	HmmWithVonMisesObservations(const size_t K, const std::vector<double> &pi, const boost::multi_array<double, 2> &A, const std::vector<double> &mus, const std::vector<double> &kappas);
	virtual ~HmmWithVonMisesObservations();

private:
	HmmWithVonMisesObservations(const HmmWithVonMisesObservations &rhs);
	HmmWithVonMisesObservations & operator=(const HmmWithVonMisesObservations &rhs);

public:
	//
	std::vector<double> & getMeanDirection()  {  return mus_;  }
	const std::vector<double> & getMeanDirection() const  {  return mus_;  }
	std::vector<double> & getConcentrationParameter()  {  return  kappas_;  }
	const std::vector<double> & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::multi_array<double, 2>::const_array_view<1>::type &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::multi_array<double, 2>::array_view<1>::type &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const boost::multi_array<double, 2> &observations, boost::multi_array<double, 2> &gamma, const double denominatorA, const size_t k);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const std::vector<boost::multi_array<double, 2> > &observationSequences, const std::vector<boost::multi_array<double, 2> > &gammas, const size_t R, const double denominatorA, const size_t k);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity();
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

private:
	std::vector<double> mus_;  // the mean directions of the von Mises distribution
	std::vector<double> kappas_;  // the concentration parameters of the von Mises distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_
