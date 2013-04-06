#if !defined(__SWL_RND_UTIL__HMM_SEGMENTER__H_)
#define __SWL_RND_UTIL__HMM_SEGMENTER__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/smart_ptr.hpp>


namespace swl {

class CDHMM;
class DDHMM;

//--------------------------------------------------------------------------
// HMM Segmenter

class SWL_RND_UTIL_API HmmSegmenter
{
public:
	//typedef HmmSegmenter base_type;
	typedef boost::numeric::ublas::vector<unsigned int> uivector_type;
	typedef boost::numeric::ublas::matrix<double> dmatrix_type;

public:
	static bool segmentByViterbiAlgorithm(const boost::scoped_ptr<DDHMM> &ddhmm, const uivector_type &observations, double &logProbability, std::size_t &startStateIndex, std::size_t &endStateIndex, uivector_type &states);
	static bool segmentByViterbiAlgorithm(const boost::scoped_ptr<CDHMM> &cdhmm, const dmatrix_type &observations, double &logProbability, std::size_t &startStateIndex, std::size_t &endStateIndex, uivector_type &states);
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HMM_SEGMENTER__H_
