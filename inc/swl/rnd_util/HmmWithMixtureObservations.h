#if !defined(__SWL_RND_UTIL__HMM_WITH_MIXTURE_OBSERVATIONS__H_)
#define __SWL_RND_UTIL__HMM_WITH_MIXTURE_OBSERVATIONS__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/multi_array.hpp>


namespace swl {

//--------------------------------------------------------------------------
// HMM with mixture observation densities

class SWL_RND_UTIL_API HmmWithMixtureObservations
{
public:
	//typedef HmmWithMixtureObservations base_type;

protected:
	HmmWithMixtureObservations(const size_t C);
	HmmWithMixtureObservations(const size_t C, const std::vector<double> &alpha);
public:
	virtual ~HmmWithMixtureObservations();

private:
	HmmWithMixtureObservations(const HmmWithMixtureObservations &rhs);
	HmmWithMixtureObservations & operator=(const HmmWithMixtureObservations &rhs);

public:
	//
	size_t getMixtureSize() const  {  return C_;  }

	std::vector<double> & getMixtureCoefficient()  {  return alpha_;  }
	const std::vector<double> & getMixtureCoefficient() const  {  return alpha_;  }

protected:
	const size_t C_;  // the number of mixture components

	std::vector<double> alpha_;  // mixture coefficients(weights)
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MIXTURE_OBSERVATIONS__H_
