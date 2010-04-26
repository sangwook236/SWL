#include "swl/Config.h"
#include "swl/util/RegionOfInterest.h"
#include "swl/math/LineSegment.h"
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/cartesian2d.hpp>
#include <boost/geometry/geometries/adapted/tuple_cartesian.hpp>
#include <boost/geometry/geometries/adapted/c_array_cartesian.hpp>
#include <boost/geometry/geometries/adapted/std_as_linestring.hpp>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
//

bool RegionOfInterest::PrComparePoints::operator()(const point_type &rhs) const
{
	const real_type eps = real_type(1.0e-15);
	return std::fabs(point_.x - rhs.x) <= eps && std::fabs(point_.y - rhs.y) <= eps;
}

#if defined(UNICODE) || defined(_UNICODE)
RegionOfInterest::RegionOfInterest(const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
RegionOfInterest::RegionOfInterest(const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: isVisible_(isVisible), color_(color), name_(name)
{
}

RegionOfInterest::RegionOfInterest(const RegionOfInterest &rhs)
: isVisible_(rhs.isVisible_), color_(rhs.color_), name_(rhs.name_)
{
}

RegionOfInterest::~RegionOfInterest()
{
}

RegionOfInterest & RegionOfInterest::operator=(const RegionOfInterest &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	isVisible_ = rhs.isVisible_;
	color_ = rhs.color_;
	name_ = rhs.name_;
	return *this;
}

bool RegionOfInterest::moveVertex(const point_type &pt1, const point_type &pt2, const real_type &vertexRadius)
{
	point_type *vertex = NULL;
	if (getNearestVertex(pt1, vertexRadius, vertex) && vertex)
	{
		*vertex += pt2 - pt1;
		return true;
	}
	else return false;
}

bool RegionOfInterest::moveVertex(const point_type &pt1, const point_type &pt2, const region_type &limitRegion, const real_type &vertexRadius)
{
	point_type *vertex = NULL;
	if (getNearestVertex(pt1, vertexRadius, vertex) && vertex)
	{
		*vertex += getMovableDistance(*vertex, pt2 - pt1, limitRegion);
		return true;
	}
	else return false;
}

bool RegionOfInterest::isNearPoint(const point_type &pt1, const point_type &pt2, const real_type &tol) const
{
	const point_type &delta = pt1 - pt2;
	return std::fabs(delta.x) <= tol && std::fabs(delta.y) <= tol;
}

RegionOfInterest::point_type RegionOfInterest::getMovableDistance(const point_type &pt, const point_type &delta, const region_type &limitRegion) const
{
	return point_type(
		(delta.x < 0) ?
			(pt.x + delta.x >= limitRegion.left ? delta.x : limitRegion.left - pt.x) :
			(pt.x + delta.x <= limitRegion.right ? delta.x : limitRegion.right - pt.x),
		(delta.y < 0) ?
			(pt.y + delta.y >= limitRegion.bottom ? delta.y : limitRegion.bottom - pt.y) :
			(pt.y + delta.y <= limitRegion.top ? delta.y : limitRegion.top - pt.y)
	);

}

RegionOfInterest::real_type RegionOfInterest::getSquareDistance(const point_type &pt1, const point_type &pt2) const
{
	return (pt2.x-pt1.x)*(pt2.x-pt1.x) + (pt2.y-pt1.y)*(pt2.y-pt1.y);
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
LineROI::LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
LineROI::LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name), pt1_(pt1), pt2_(pt2)
{
}

LineROI::LineROI(const LineROI &rhs)
: base_type(rhs), pt1_(rhs.pt1_), pt2_(rhs.pt2_)
{
}

LineROI::~LineROI()
{
}

LineROI & LineROI::operator=(const LineROI &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	pt1_ = rhs.pt1_;
	pt2_ = rhs.pt2_;
	return *this;
}

void LineROI::moveRegion(const point_type &pt1, const point_type &pt2)
{
	const point_type &delta = pt2 - pt1;

	pt1_ += delta;
	pt2_ += delta;
}

void LineROI::moveRegion(const point_type &pt1, const point_type &pt2, const region_type &limitRegion)
{
	const point_type &delta = pt2 - pt1;
	const region_type rgn(pt1_, pt2_);
	const point_type disp(
		(delta.x < 0) ?
			(rgn.left + delta.x >= limitRegion.left ? delta.x : limitRegion.left - rgn.left) :
			(rgn.right + delta.x <= limitRegion.right ? delta.x : limitRegion.right - rgn.right),
		(delta.y < 0) ?
			(rgn.bottom + delta.y >= limitRegion.bottom ? delta.y : limitRegion.bottom - rgn.bottom) :
			(rgn.top + delta.y <= limitRegion.top ? delta.y : limitRegion.top - rgn.top)
	);

	pt1_ += disp;
	pt2_ += disp;
}

bool LineROI::isVertex(const point_type &pt, const real_type &radius) const
{
	return isNearPoint(pt1_, pt, radius) || isNearPoint(pt2_, pt, radius);
}

bool LineROI::include(const point_type &pt, const real_type &tol) const
{
	return LineSegment2<real_type>(pt1_, pt2_).include(pt, tol);
}

bool LineROI::getNearestVertex(const point_type &pt, const real_type &radius, point_type *&vertex)
{
	const real_type &dist1 = getSquareDistance(pt, pt1_);
	const real_type &dist2 = getSquareDistance(pt, pt2_);
	if (dist1 <= dist2)
	{
		if (dist1 < radius*radius)
		{
			vertex = &pt1_;
			return true;
		}
		else
		{
			vertex = NULL;
			return false;
		}
	}
	else
	{
		if (dist2 < radius*radius)
		{
			vertex = &pt2_;
			return true;
		}
		else
		{
			vertex = NULL;
			return false;
		}
	}
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
RectangleROI::RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
RectangleROI::RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name), rect_(pt1, pt2)
{
}

RectangleROI::RectangleROI(const RectangleROI &rhs)
: base_type(rhs), rect_(rhs.rect_)
{
}

RectangleROI::~RectangleROI()
{
}

RectangleROI & RectangleROI::operator=(const RectangleROI &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	rect_ = rhs.rect_;
	return *this;
}

void RectangleROI::moveRegion(const point_type &pt1, const point_type &pt2)
{
	rect_ += pt2 - pt1;
}

void RectangleROI::moveRegion(const point_type &pt1, const point_type &pt2, const region_type &limitRegion)
{
	const point_type &delta = pt2 - pt1;
	const point_type disp(
		(delta.x < 0) ?
			(rect_.left + delta.x >= limitRegion.left ? delta.x : limitRegion.left - rect_.left) :
			(rect_.right + delta.x <= limitRegion.right ? delta.x : limitRegion.right - rect_.right),
		(delta.y < 0) ?
			(rect_.bottom + delta.y >= limitRegion.bottom ? delta.y : limitRegion.bottom - rect_.bottom) :
			(rect_.top + delta.y <= limitRegion.top ? delta.y : limitRegion.top - rect_.top)
	);

	rect_ += disp;
}

bool RectangleROI::isVertex(const point_type &pt, const real_type &radius) const
{
	return isNearPoint(point_type(rect_.left, rect_.top), pt, radius) || isNearPoint(point_type(rect_.left, rect_.bottom), pt, radius) ||
		isNearPoint(point_type(rect_.right, rect_.top), pt, radius) || isNearPoint(point_type(rect_.right, rect_.bottom), pt, radius);
}

bool RectangleROI::include(const point_type &pt, const real_type &tol) const
{
	return rect_.left-tol <= pt.x && pt.x <= rect_.right+tol && rect_.bottom-tol <= pt.y && pt.y <= rect_.top+tol;
}

bool RectangleROI::getNearestVertex(const point_type &pt, const real_type &radius, point_type *&vertex)
{
	// TODO [add] >>
	throw std::runtime_error("not yet implemented");
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
ROIWithVariablePoints::ROIWithVariablePoints(const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
ROIWithVariablePoints::ROIWithVariablePoints(const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name), points_()
{
}

#if defined(UNICODE) || defined(_UNICODE)
ROIWithVariablePoints::ROIWithVariablePoints(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
ROIWithVariablePoints::ROIWithVariablePoints(const points_type &points, const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name), points_(points)
{
}

ROIWithVariablePoints::ROIWithVariablePoints(const ROIWithVariablePoints &rhs)
: base_type(rhs), points_(rhs.points_)
{
}

ROIWithVariablePoints::~ROIWithVariablePoints()
{
}

ROIWithVariablePoints & ROIWithVariablePoints::operator=(const ROIWithVariablePoints &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	points_.assign(rhs.points_.begin(), rhs.points_.end());
	return *this;
}

void ROIWithVariablePoints::addPoint(const point_type &point)
{
	points_.push_back(point);
}

void ROIWithVariablePoints::removePoint(const point_type &point)
{
	points_.remove_if(PrComparePoints(point));
}

void ROIWithVariablePoints::moveRegion(const point_type &pt1, const point_type &pt2)
{
	const point_type &delta = pt2 - pt1;

	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
		*it += delta;
}

void ROIWithVariablePoints::moveRegion(const point_type &pt1, const point_type &pt2, const region_type &limitRegion)
{
	// TODO [add] >>
	throw std::runtime_error("not yet implemented");
}

bool ROIWithVariablePoints::isVertex(const point_type &pt, const real_type &radius) const
{
	for (points_type::const_iterator it = points_.begin(); it != points_.end(); ++it)
		if (isNearPoint(*it, pt, radius)) return true;
	return false;
}

bool ROIWithVariablePoints::getNearestVertex(const point_type &pt, const real_type &radius, point_type *&vertex)
{
	vertex = NULL;

	real_type minDist = std::numeric_limits<real_type>::max();
	const real_type radius2 = radius * radius;
	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
	{
		const real_type &dist = getSquareDistance(*it, pt);
		if (dist < radius2 && dist < minDist)
		{
			// TODO [check] >>
			vertex = &(*it);
			minDist = dist;
		}
	}

	return NULL != vertex;
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
PolylineROI::PolylineROI(const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
PolylineROI::PolylineROI(const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name)
{
}

#if defined(UNICODE) || defined(_UNICODE)
PolylineROI::PolylineROI(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
PolylineROI::PolylineROI(const points_type &points, const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(points, isVisible, color, name)
{
}

PolylineROI::PolylineROI(const PolylineROI &rhs)
: base_type(rhs)
{
}

PolylineROI::~PolylineROI()
{
}

PolylineROI & PolylineROI::operator=(const PolylineROI &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	return *this;
}

bool PolylineROI::include(const point_type &pt, const real_type &tol) const
{
	if (points_.size() < 3) return false;

	boost::geometry::polygon_2d poly;
	boost::geometry::ring_type<boost::geometry::polygon_2d>::type &ring = exterior_ring(poly);
	for (points_type::const_iterator it = points_.begin(); it != points_.end(); ++it)
		append(ring, boost::geometry::make<boost::geometry::point_2d>(it->x, it->y));
	// TODO [check] >>
	append(ring, boost::geometry::make<boost::geometry::point_2d>(points_.front().x, points_.front().y));

	// TODO [check] >>
	correct(poly);

	boost::geometry::polygon_2d hull;
	convex_hull(poly, hull);

	return within(boost::geometry::make<boost::geometry::point_2d>(pt.x, pt.y), hull);
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
PolygonROI::PolygonROI(const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
PolygonROI::PolygonROI(const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, color, name)
{
}

#if defined(UNICODE) || defined(_UNICODE)
PolygonROI::PolygonROI(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name /*= std::wstring()*/)
#else
PolygonROI::PolygonROI(const points_type &points, const bool isVisible, const color_type &color, const std::string &name /*= std::string()*/)
#endif
: base_type(points, isVisible, color, name)
{
}

PolygonROI::PolygonROI(const PolygonROI &rhs)
: base_type(rhs)
{
}

PolygonROI::~PolygonROI()
{
}

PolygonROI & PolygonROI::operator=(const PolygonROI &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	return *this;
}

bool PolygonROI::include(const point_type &pt, const real_type &tol) const
{
	points_type::const_iterator itPrev = points_.begin();
	points_type::const_iterator it = itPrev;
	++it;
	if (points_.end() == itPrev || points_.end() == it) return false;

	for (; it != points_.end(); ++it)
	{
		if (LineSegment2<real_type>(*itPrev, *it).include(pt, tol)) return true;
		itPrev = it;
	}
	return false;
}

}  // namespace swl
