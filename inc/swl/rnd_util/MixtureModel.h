#if !defined(__SWL_RND_UTIL__MIXTURE_MODEL__H_)
#define __SWL_RND_UTIL__MIXTURE_MODEL__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
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
	MixtureModel(const size_t K, const size_t D);
	MixtureModel(const size_t K, const size_t D, const std::vector<double> &pi);
	virtual ~MixtureModel();

private:
	MixtureModel(const MixtureModel &rhs);  // not implemented
	MixtureModel & operator=(const MixtureModel &rhs);  // not implemented

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

	unsigned int generateState() const;

protected:
	const size_t K_;  // the number of mixture components
	const size_t D_;  // the dimension of observation symbols

	std::vector<double> pi_;  // mixture coefficients(weights)
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__MIXTURE_MODEL__H_
