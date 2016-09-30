#if !defined(__SWL_MATH__GEOMETRY_UTIL__H_)
#define __SWL_MATH__GEOMETRY_UTIL__H_ 1


#include "swl/math/ExportMath.h"
#include "swl/base/Point.h"
#include <list>


namespace swl {

//-----------------------------------------------------------------------------------------
// Geometry Util.

struct SWL_MATH_API GeometryUtil
{
public:
	// Compute the nearest point.
	/// Line equation: a * x + b * y + c = 0.
	static bool computeNearestPointWithLine(const double x0, const double y0, const double a, const double b, const double c, double& nearestX, double& nearestY);
	/// Plane equation: a * x + b * y + c * z + d = 0.
	static bool computeNearestPointWithPlane(const double x0, const double y0, const double z0, const double a, const double b, const double c, const double d, double& nearestX, double& nearestY, double& nearestZ);
	/// Quadratic equation: a * x^2 + b * x + c * y + d = 0.
	static bool computeNearestPointWithQuadratic(const double x0, const double y0, const double a, const double b, const double c, const double d, double& nearestX, double& nearestY);

	//
	static void getConvexHull(const std::list<Point2<float> > &points, std::list<Point2<float> > &convexHull);
	static bool within(const Point2<float> &pt, const std::list<Point2<float> > &points, const float tol);
	static bool withinConvexHull(const Point2<float> &pt, const std::list<Point2<float> > &convexHull, const float tol);
};

}  // namespace swl


#endif  // __SWL_MATH__GEOMETRY_UTIL__H_
