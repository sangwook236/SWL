#include "swl/Config.h"
#include "swl/math/GeometryUtil.h"
#include "swl/math/LineSegment.h"
#if 0
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/cartesian2d.hpp>
#else
#include "ConvexHull.h"
#endif
#include <vector>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {

// line equation: (x - x0) / a = (y - y0) / b ==> b * (x - x0) - a * (y - y0) = 0
bool isInTheSameSide(const double &px, const double &py, const double &cx, const double &cy, const double &a, const double &b, const double &x0, const double &y0, const double &tol)
{
	return (b * (px - x0) - a * (py - y0)) * (b * (cx - x0) - a * (cy - y0)) >= 0.0;
}

}

/*static*/ bool GeometryUtil::within(const Point2<float> &pt, const std::list<Point2<float> > &points, const float tol)
{
	if (points.size() < 3) return false;

#if 0
	boost::geometry::polygon_2d poly;
	boost::geometry::ring_type<boost::geometry::polygon_2d>::type &ring = exterior_ring(poly);
	for (std::list<Point2<float> >::const_iterator it = points.begin(); it != points.end(); ++it)
		append(ring, boost::geometry::make<boost::geometry::point_2d>(it->x, it->y));
	append(ring, boost::geometry::make<boost::geometry::point_2d>(points.front().x, points.front().y));

	// TODO [check] >>
	correct(poly);

	boost::geometry::polygon_2d hull;
	convex_hull(poly, hull);

	return within(boost::geometry::make<boost::geometry::point_2d>(pt.x, pt.y), hull);  // not boundary point, but internal point
#else
	std::vector<point2d> pts;
	std::vector<point2d> convex_hull;

	for (std::list<Point2<float> >::const_iterator it = points.begin(); it != points.end(); ++it)
		pts.push_back(point2d(it->x, it->y));

	GrahamScanConvexHull()(pts, convex_hull);

	const size_t count = convex_hull.size();
	if (count < 2) return false;
	else if (2 == count)
	{
		const point2d &pt1 = convex_hull.front();
		const point2d &pt2 = convex_hull.back();

		return LineSegment2<float>(Point2<float>((float)pt1.x, (float)pt1.y), Point2<float>((float)pt2.x, (float)pt2.y)).include(pt, tol);
	}
	else
	{
		point2d center(0, 0);
		for (std::vector<point2d>::iterator it = convex_hull.begin(); it != convex_hull.end(); ++it)
		{
			center.x += it->x;
			center.y += it->y;
		}
		center.x /= (double)count;
		center.y /= (double)count;

		std::vector<point2d>::iterator itPrev = convex_hull.begin();
		std::vector<point2d>::iterator it = itPrev;
		++it;
		//if (convex_hull.end() == itPrev || convex_hull.end() == it) return false;

		for (; it != convex_hull.end(); ++it)
		{
			if (!isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol)) return false;
			itPrev = it;
		}
		it = convex_hull.begin();
		return isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol);
	}
#endif
}

}  //  namespace swl
