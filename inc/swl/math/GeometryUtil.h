#if !defined(__SWL_MATH__GEOMETRY_UTIL__H_)
#define __SWL_MATH__GEOMETRY_UTIL__H_ 1


#include "swl/math/ExportMath.h"
#include "swl/base/Point.h"
#include <list>


namespace swl {

//-----------------------------------------------------------------------------------------
// struct GeometryUtil

struct SWL_MATH_API GeometryUtil
{
public:
	///
	static bool within(const Point2<float> &pt, const std::list<Point2<float> > &points, const float tol);
};

}  // namespace swl


#endif  // __SWL_MATH__GEOMETRY_UTIL__H_
