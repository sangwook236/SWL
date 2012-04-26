#if !defined(__SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include <boost/smart_ptr.hpp>


namespace swl {

struct VonMisesTargetDistribution;
struct UnivariateNormalProposalDistribution;
struct UnivariateUniformProposalDistribution;

//--------------------------------------------------------------------------
// continuous density HMM with von Mises observation densities

class SWL_RND_UTIL_API HmmWithVonMisesObservations: public CDHMM
{
public:
	typedef CDHMM base_type;
	typedef boost::numeric::ublas::vector<double> dvector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<unsigned int> uimatrix_type;

public:
	HmmWithVonMisesObservations(const size_t K);
	HmmWithVonMisesObservations(const size_t K, const dvector_type &pi, const dmatrix_type &A, const dvector_type &mus, const dvector_type &kappas);
	virtual ~HmmWithVonMisesObservations();

private:
	HmmWithVonMisesObservations(const HmmWithVonMisesObservations &rhs);  // not implemented
	HmmWithVonMisesObservations & operator=(const HmmWithVonMisesObservations &rhs);  // not implemented

public:
	//
	dvector_type & getMeanDirection()  {  return mus_;  }
	const dvector_type & getMeanDirection() const  {  return mus_;  }
	dvector_type & getConcentrationParameter()  {  return  kappas_;  }
	const dvector_type & getConcentrationParameter() const  {  return  kappas_;  }

protected:
	// if state == 0, hidden state = [ 1 0 0 ... 0 0 ]
	// if state == 1, hidden state = [ 0 1 0 ... 0 0 ]
	// ...
	// if state == N-1, hidden state = [ 0 0 0 ... 0 1 ]
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const boost::numeric::ublas::matrix_row<const dmatrix_type> &observation) const;
	// if seed != -1, the seed value is set
	/*virtual*/ void doGenerateObservationsSymbol(const unsigned int state, boost::numeric::ublas::matrix_row<dmatrix_type> &observation, const unsigned int seed = (unsigned int)-1) const;

	// for a single independent observation sequence
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const size_t N, const unsigned int state, const dmatrix_type &observations, dmatrix_type &gamma, const double denominatorA);
	// for multiple independent observation sequences
	/*virtual*/ void doEstimateObservationDensityParametersInMStep(const std::vector<size_t> &Ns, const unsigned int state, const std::vector<dmatrix_type> &observationSequences, const std::vector<dmatrix_type> &gammas, const size_t R, const double denominatorA);

	//
	/*virtual*/ bool doReadObservationDensity(std::istream &stream);
	/*virtual*/ bool doWriteObservationDensity(std::ostream &stream) const;
	/*virtual*/ void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	/*virtual*/ void doNormalizeObservationDensityParameters()
	{
		// do nothing
	}

private:
	dvector_type mus_;  // the mean directions of the von Mises distribution. 0 <= mu < 2 * pi. [rad].
	dvector_type kappas_;  // the concentration parameters of the von Mises distribution. kappa >= 0.

	mutable boost::scoped_ptr<VonMisesTargetDistribution> targetDist_;
#if 0
	mutable boost::scoped_ptr<UnivariateNormalProposalDistribution> proposalDist_;
#else
	mutable boost::scoped_ptr<UnivariateUniformProposalDistribution> proposalDist_;
#endif
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_VON_MISES_OBSERVATIONS__H_
