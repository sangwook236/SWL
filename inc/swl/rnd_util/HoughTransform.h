#if !defined(__SWL_RND_UTIL__HOUGH_TRANSFORM__H_)
#define __SWL_RND_UTIL__HOUGH_TRANSFORM__H_ 1


#include "swl/rnd_util/ExportRndUtil.h"
#include <map>
#include <set>


namespace swl {

//--------------------------------------------------------------------------
//

class SWL_RND_UTIL_API GeneralizedHoughTransform
{
public:
	//typedef GeneralizedHoughTransform base_type;
	typedef std::set<std::size_t> rtable_entry_type;
	typedef std::map<std::size_t, rtable_entry_type> rtable_type;

protected:
	GeneralizedHoughTransform(const std::size_t tangentAngleCount);
public:
	virtual ~GeneralizedHoughTransform();

public:
	virtual bool run();

protected:
	const std::size_t tangentAngleCount_;  // determine a resolution of tangent angles

	// { an index of a tangent angle at a reference point (x,y), a set of indices which indicate a pair (distance,angle) or (dx,dy) }
	// 0 <= a tangent angle < 2 * pi
	// 0 <= an index of a tangent angle < tangentAngleCount
	rtable_type rTable_;
};

}  // namespace swl


#endif  // __SWL_RND_UTIL__HOUGH_TRANSFORM__H_
