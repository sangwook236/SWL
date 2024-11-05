#if !defined(__SWL_RND_UTIL__CDHMM_WITH_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__CDHMM_WITH_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/CDHMM.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/smart_ptr.hpp>
#include <vector>


namespace swl {

//--------------------------------------------------------------------------
// CDHMM with mixture observation densities

class SWL_RND_UTIL_API CDHMMWithMixtureObservations: public CDHMM
{
public:
	typedef CDHMM base_type;

protected:
	CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C);  // for ML learning.
	CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type &pi, const dmatrix_type &A, const dmatrix_type &alphas);
	CDHMMWithMixtureObservations(const size_t K, const size_t D, const size_t C, const dvector_type *pi_conj, const dmatrix_type *A_conj, const dmatrix_type *alphas_conj);  // for MAP learning using conjugate prior.
public:
	virtual ~CDHMMWithMixtureObservations();

private:
	CDHMMWithMixtureObservations(const CDHMMWithMixtureObservations &rhs);  // not implemented.
	CDHMMWithMixtureObservations & operator=(const CDHMMWithMixtureObservations &rhs);  // not implemented.

public:
	//
	size_t getMixtureSize() const  {  return C_;  }

	dmatrix_type & getMixtureCoefficient()  {  return alphas_;  }
	const dmatrix_type & getMixtureCoefficient() const  {  return alphas_;  }

protected:
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const dvector_type &observation) const;
	/*virtual*/ double doEvaluateEmissionProbability(const unsigned int state, const size_t n, const dmatrix_type &observations) const;
	virtual double doEvaluateEmissionMixtureComponentProbability(const unsigned int state, const unsigned int component, const dvector_type &observation) const = 0;
	virtual double doEvaluateEmissionMixtureComponentProbability(const unsigned int state, const unsigned int component, const size_t n, const dmatrix_type &observations) const;

	void normalizeObservationDensityParameters(const size_t K);

protected:
	const size_t C_;  // the number of mixture components.

	dmatrix_type alphas_;  // mixture coefficients(weights).

	// hyperparameters for the conjugate prior.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	boost::scoped_ptr<const dmatrix_type> alphas_conj_;  // eta.
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__CDHMM_WITH_MIXTURE_OBSERVATIONS__H_
