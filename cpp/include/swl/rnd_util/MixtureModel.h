#if !defined(__SWL_RND_UTIL__MIXTURE_MODEL__H_)
#define __SWL_RND_UTIL__MIXTURE_MODEL__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/smart_ptr.hpp>
#include <vector>
#include <iosfwd>


namespace swl {

//--------------------------------------------------------------------------
// mixture model

class SWL_RND_UTIL_API MixtureModel
{
public:
	//typedef MixtureModel base_type;

protected:
	MixtureModel(const size_t K, const size_t D);  // for ML learning.
	MixtureModel(const size_t K, const size_t D, const std::vector<double> &pi);
	MixtureModel(const size_t K, const size_t D, const std::vector<double> *pi_conj);  // for MAP learning using conjugate prior.
	virtual ~MixtureModel();

private:
	MixtureModel(const MixtureModel &rhs);  // not implemented.
	MixtureModel & operator=(const MixtureModel &rhs);  // not implemented.

public:
	bool readModel(std::istream &stream);
	bool writeModel(std::ostream &stream) const;

	void initializeModel(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity);
	void normalizeModelParameters();

	//
	size_t getMixtureSize() const  {  return K_;  }
	size_t getObservationDim() const  {  return D_;  }

	std::vector<double> & getMixtureCoefficient()  {  return pi_;  }
	const std::vector<double> & getMixtureCoefficient() const  {  return pi_;  }

protected:
	virtual bool doReadObservationDensity(std::istream &stream) = 0;
	virtual bool doWriteObservationDensity(std::ostream &stream) const = 0;
	virtual void doInitializeObservationDensity(const std::vector<double> &lowerBoundsOfObservationDensity, const std::vector<double> &upperBoundsOfObservationDensity) = 0;
	virtual void doNormalizeObservationDensityParameters() = 0;

	virtual void doInitializeRandomSampleGeneration(const unsigned int seed = (unsigned int)-1) const
	{
		// do nothing.
	}
	virtual void doFinalizeRandomSampleGeneration() const
	{
		// do nothing.
	}

	virtual bool doDoHyperparametersOfConjugatePriorExist() const
	{  return NULL != pi_conj_.get();  }

	unsigned int generateState() const;

protected:
	const size_t K_;  // the number of mixture components.
	const size_t D_;  // the dimension of observation symbols.

	std::vector<double> pi_;  // mixture coefficients(weights).

	// hyperparameters for the conjugate prior.
	//	[ref] "Maximum a Posteriori Estimation for Multivariate Gaussian Mixture Observations of Markov Chains", J.-L. Gauvain adn C.-H. Lee, TSAP, 1994.
	//	[ref] "Pattern Recognition and Machine Learning", C. M. Bishop, Springer, 2006.
	//	[ref] "EM Algorithm 3 - THE EM Algorithm for MAP Estimates of HMM", personal note.
	boost::scoped_ptr<const std::vector<double> > pi_conj_;  // for mixture coefficients(weights).
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__MIXTURE_MODEL__H_
