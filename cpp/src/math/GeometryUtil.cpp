#include "swl/Config.h"
#include "swl/math/GeometryUtil.h"
#include "swl/math/LineSegment.h"
#if 0
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/cartesian2d.hpp>
#else
#include "ConvexHull.h"
#endif
#include <gsl/gsl_poly.h>
#include <vector>
#include <cassert>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

namespace {
namespace local {

// line equation: (x - x0) / a = (y - y0) / b ==> b * (x - x0) - a * (y - y0) = 0
bool isInTheSameSide(const double &px, const double &py, const double &cx, const double &cy, const double &a, const double &b, const double &x0, const double &y0, const double &tol)
{
	return (b * (px - x0) - a * (py - y0)) * (b * (cx - x0) - a * (cy - y0)) >= 0.0;
}

}  // namespace local
}  // unnamed namespace

// Line equation: a * x + b * y + c = 0.
/*static*/ bool GeometryUtil::computeNearestPointWithLine(const double x0, const double y0, const double a, const double b, const double c, double& nearestX, double& nearestY)
{
	const double eps = 1.0e-20;

	if (std::abs(a) < eps && std::abs(b) < eps) return false;

#if 0
	// TODO [decide] >> How to decide P = (xp, yp).
	const double xp = 0.0, yp = -c / b;  // An arbitrary point on the line.
										 //const double dx = b, dy = -a;  // Directional vector.

										 //const double k = ((x0 - xp) * dx + (y0 - yp) * dy) / (dx*dx + dy*dy);
										 //nearestX = xp + k * dx;
										 //nearestY = yp + k * dy;
	const double k = ((x0 - xp) * b - (y0 - yp) * a) / (a*a + b*b);
	nearestX = xp + k * b;
	nearestY = yp - k * a;
#else
	const double denom = a*a + b*b, d = a * y0 - b * x0;
	nearestX = -(a * c + b * d) / denom;
	nearestY = (a * d - b * c) / denom;
#endif

	return true;
}

// Plane equation: a * x + b * y + c * z + d = 0.
/*static*/ bool GeometryUtil::computeNearestPointWithPlane(const double x0, const double y0, const double z0, const double a, const double b, const double c, const double d, double& nearestX, double& nearestY, double& nearestZ)
{
	const double eps = 1.0e-20;

	if (std::abs(a) < eps && std::abs(b) < eps && std::abs(c) < eps) return false;

#if 0
	// TODO [decide] >> How to decide P = (xp, yp, zp).
	const double xp = 0.0, yp = 0.0, zp = -d / c;  // An arbitrary point on the plane.
												   //const double nx = a, ny = b, nz = c;  // Normal vector.

												   //const double k = ((xp - x0) * nx + (yp - y0) * ny + (zp - z0) * nz) / (nx*nx + ny*ny + nz*nz);
												   //nearestX = x0 + k * nx;
												   //nearestY = y0 + k * ny;
												   //nearestZ = z0 + k * nz;
	const double k = ((xp - x0) * a + (yp - y0) * b + (zp - z0) * c) / (a*a + b*b + c*c);
	nearestX = x0 + k * a;
	nearestY = y0 + k * b;
	nearestZ = z0 + k * c;
#else
	const double t = (-a*x0 - b*y0 - c*z0 - d) / (a*a + b*b + c*c);
	nearestX = a * t + x0;
	nearestY = b * t + y0;
	nearestZ = c * t + z0;
#endif

	return true;
}

// Quadratic equation: a * x^2 + b * x + c * y + d = 0.
/*static*/ bool GeometryUtil::computeNearestPointWithQuadratic(const double x0, const double y0, const double a, const double b, const double c, const double d, double& nearestX, double& nearestY)
{
	const double eps = 1.0e-20;

	if (std::abs(a) < eps)  // Linear equation if a = 0.
		return computeNearestPointWithLine(x0, y0, b, c, d, nearestX, nearestY);
	else  // Quadratic equation.
	{
		const double c2 = c * c;
		if (c2 < eps) return false;

		const double aa = 4.0*a*a / c2, bb = 6.0*a*b / c2, cc = 2.0*(b*b / c2 + 2.0*a*(d + c * y0) / c2 + 1.0), dd = 2.0*(b*(d + c * y0) / c2 - x0);

		double roots[3] = { 0.0, };
		switch (gsl_poly_solve_cubic(bb / aa, cc / aa, dd / aa, &roots[0], &roots[1], &roots[2]))
		{
		case 1:
			nearestX = roots[0], nearestY = -(a * nearestX*nearestX + b * nearestX + d) / c;
			return true;
		case 3:
		{
			double minDist2 = std::numeric_limits<double>::max();
			for (int i = 0; i < 3; ++i)
			{
				const double xx = roots[i], yy = -(a * xx*xx + b * xx + d) / c;
				const double dist2 = (xx - x0)*(xx - x0) + (yy - y0)*(yy - y0);
				if (dist2 < minDist2)
				{
					minDist2 = dist2;
					nearestX = xx;
					nearestY = yy;
				}
			}
		}
		return true;
		default:
			assert(false);
			return false;
		}
	}

	return false;
}

/*static*/ void GeometryUtil::getConvexHull(const std::list<Point2<float> > &points, std::list<Point2<float> > &convexHull)
{
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

	// FIXME [fix] >>
	const boost::geometry::polygon_2d::ring_type &outer_ring = hull.outer();
	for (boost::geometry::polygon_2d::iterator it = hull.begin(); it != hull.end(); ++it)
		convexHull.push_back(Point2<float>(it->x, it->y));
#else
	std::vector<point2d> pts;
	std::vector<point2d> hull;

	for (std::list<Point2<float> >::const_iterator it = points.begin(); it != points.end(); ++it)
		pts.push_back(point2d(it->x, it->y));

	GrahamScanConvexHull()(pts, hull);

	for (std::vector<point2d>::iterator it = hull.begin(); it != hull.end(); ++it)
		convexHull.push_back(Point2<float>((float)it->x, (float)it->y));
#endif
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
	std::vector<point2d> hull;

	for (std::list<Point2<float> >::const_iterator it = points.begin(); it != points.end(); ++it)
		pts.push_back(point2d(it->x, it->y));

	GrahamScanConvexHull()(pts, hull);

	const size_t count = hull.size();
	if (count < 2) return false;
	else if (2 == count)
	{
		const point2d &pt1 = hull.front();
		const point2d &pt2 = hull.back();

		return LineSegment2<float>(Point2<float>((float)pt1.x, (float)pt1.y), Point2<float>((float)pt2.x, (float)pt2.y)).include(pt, tol);
	}
	else
	{
		point2d center(0, 0);
		for (std::vector<point2d>::iterator it = hull.begin(); it != hull.end(); ++it)
		{
			center.x += it->x;
			center.y += it->y;
		}
		center.x /= (double)count;
		center.y /= (double)count;

		std::vector<point2d>::iterator itPrev = hull.begin(), it = itPrev;
		++it;
		//if (hull.end() == itPrev || hull.end() == it) return false;

		for (; it != hull.end(); ++it)
		{
			if (!local::isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol)) return false;
			itPrev = it;
		}
		it = hull.begin();
		return local::isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol);
	}
#endif
}

/*static*/ bool GeometryUtil::withinConvexHull(const Point2<float> &pt, const std::list<Point2<float> > &convexHull, const float tol)
{
	const size_t count = convexHull.size();
	if (count < 2) return false;
	else if (2 == count)
		return LineSegment2<float>(convexHull.front(), convexHull.back()).include(pt, tol);
	else
	{
		Point2<float> center(0, 0);
		for (std::list<Point2<float> >::const_iterator it = convexHull.begin(); it != convexHull.end(); ++it)
		{
			center.x += it->x;
			center.y += it->y;
		}
		center.x /= (float)count;
		center.y /= (float)count;

		std::list<Point2<float> >::const_iterator itPrev = convexHull.begin(), it = itPrev;
		++it;
		//if (convexHull.end() == itPrev || convexHull.end() == it) return false;

		for (; it != convexHull.end(); ++it)
		{
			if (!local::isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol)) return false;
			itPrev = it;
		}
		it = convexHull.begin();
		return local::isInTheSameSide(pt.x, pt.y, center.x, center.y, it->x - itPrev->x, it->y - itPrev->y, itPrev->x, itPrev->y, tol);
	}
}

}  //  namespace swl
