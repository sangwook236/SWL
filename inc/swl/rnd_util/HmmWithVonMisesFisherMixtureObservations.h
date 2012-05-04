#if !defined(__SWL_RND_UTIL__HMM_WITH_VON_MISES_FISHER_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_VON_MISES_FISHER_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include "swl/rnd_util/HmmWithMixtureObservations.h"
#include <boost/multi_array.hpp>


namespace swl {

//--------------------------------------------------------------------------
// continuous density HMM with von Mises-Fisher mixture observation densities

class SWL_RND_UTIL_API HmmWithVonMisesFisherMixtureObservations: public CDHMM, HmmWithMixtureObservations
{
public:
	typedef CDHMM base_type;
	//typedef HmmWithMixtureObservations base_type;
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<unsigned int> uimatrix_type;

public:
	HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C);
	HmmWithVonMisesFisherMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas, const boost::multi_array<dvector_type, 2> &mus, const dmatrix_type &kappas);
	virtual ~HmmWithVonMisesFisherMixtureObservations();

private:
	HmmWithVonMisesFisherMixtureObservations(const HmmWithVonMisesFisherMixtureObservations &rhs);  // not implemented
	HmmWithVonMisesFisherMixtureObservations & operator=(const HmmWithVonMisesFisherMixtureObservations &rhs);  // not implemented

public:
	//
	boost::multi_array<dvector_type, 2> & getMeanDirection()  {  return mus_;  }
	const boost::multi_array<dvector_type, 2> & getMeanDirection() const  {  return mus_;  }
	dmatrix_type & getConcentrationParameter()  {  return  kappas_;  }
	const dmatrix_type & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersByML(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersByML(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		HmmWithMixtureObservations::normalizeObservationDensityParameters(K_);
	}

private:
	boost::multi_array<dvector_type, 2> mus_;  // the sets of mean vectors of each components in the von Mises-Fisher mixture distribution
	dmatrix_type kappas_;  // the sets of concentration parameters of each components in the von Mises-Fisher mixture distribution
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_VON_MISES_FISHER_MIXTURE_OBSERVATIONS__H_
