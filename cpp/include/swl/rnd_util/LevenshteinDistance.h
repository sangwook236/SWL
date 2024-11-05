#if !defined(__SWL_RND_UTIL__LEVENSHTEIN_DISTANCE__H_)
#define __SWL_RND_UTIL__LEVENSHTEIN_DISTANCE__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <string>


namespace swl {

SWL_RND_UTIL_API std::size_t computeLevenshteinDistance(const std::string &s1, const std::string &s2);

}  // namespace swl


#endif  // __SWL_RND_UTIL__LEVENSHTEIN_DISTANCE__H_
