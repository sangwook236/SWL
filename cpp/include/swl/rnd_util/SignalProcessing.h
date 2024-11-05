#if !defined(__SWL_RND_UTIL__SIGNAL_PROCESSING__H_)
#define __SWL_RND_UTIL__SIGNAL_PROCESSING__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <vector>


namespace swl {

//--------------------------------------------------------------------------
// Signal processing.

class SWL_RND_UTIL_API SignalProcessing
{
public:
	static void filter(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &x, std::vector<double> &y);
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__SIGNAL_PROCESSING__H_
