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
	HmmWithMixtureObservations(const size_t C, const size_t K);
	HmmWithMixtureObservations(const size_t C, const size_t K, const boost::multi_array<double, 2> &alphas);
public:
	virtual ~HmmWithMixtureObservations();

private:
	HmmWithMixtureObservations(const HmmWithMixtureObservations &rhs);
	HmmWithMixtureObservations & operator=(const HmmWithMixtureObservations &rhs);

public:
	//
	size_t getMixtureSize() const  {  return C_;  }

	boost::multi_array<double, 2> & getMixtureCoefficient()  {  return alphas_;  }
	const boost::multi_array<double, 2> & getMixtureCoefficient() const  {  return alphas_;  }

protected:
	void normalizeObservationDensityParameters(const size_t K);

protected:
	const size_t C_;  // the number of mixture components

	boost::multi_array<double, 2> alphas_;  // mixture coefficients(weights)
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_WITH_MIXTURE_OBSERVATIONS__H_
