#include "swl/Config.h"
#include "swl/util/RegionOfInterest.h"
#include "swl/math/LineSegment.h"
#include "swl/math/GeometryUtil.h"
#include <vector>
#include <algorithm>
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
RegionOfInterest::RegionOfInterest(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
RegionOfInterest::RegionOfInterest(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: isVisible_(isVisible), lineWidth_(lineWidth), pointSize_(pointSize), lineColor_(lineColor), pointColor_(pointColor), name_(name)
{
}

RegionOfInterest::RegionOfInterest(const RegionOfInterest &rhs)
: isVisible_(rhs.isVisible_), lineWidth_(rhs.lineWidth_), pointSize_(rhs.pointSize_), lineColor_(rhs.lineColor_), pointColor_(rhs.pointColor_), name_(rhs.name_)
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
	lineWidth_ = rhs.lineWidth_;
	lineColor_ = rhs.lineColor_;
	pointSize_ = rhs.pointSize_;
	pointColor_ = rhs.pointColor_;
	name_ = rhs.name_;
	return *this;
}

/*static*/ bool RegionOfInterest::isNearPoint(const point_type &pt1, const point_type &pt2, const real_type &tol)
{
	const point_type &delta = pt1 - pt2;
	return std::fabs(delta.x) <= tol && std::fabs(delta.y) <= tol;
}

/*static*/ RegionOfInterest::point_type RegionOfInterest::getMovableDistance(const point_type &pt, const point_type &delta, const region_type &limitRegion)
{
	return point_type(
		(delta.x < 0) ?
			(pt.x <= limitRegion.left ? real_type(0) : (pt.x + delta.x >= limitRegion.left ? delta.x : limitRegion.left - pt.x)) :
			(pt.x >= limitRegion.right ? real_type(0) : (pt.x + delta.x <= limitRegion.right ? delta.x : limitRegion.right - pt.x)),
		(delta.y < 0) ?
			(pt.y <= limitRegion.bottom ? real_type(0) : (pt.y + delta.y >= limitRegion.bottom ? delta.y : limitRegion.bottom - pt.y)) :
			(pt.y >= limitRegion.top ? real_type(0) : (pt.y + delta.y <= limitRegion.top ? delta.y : limitRegion.top - pt.y))
	);
}

/*static*/ RegionOfInterest::point_type RegionOfInterest::getMovableDistance(const region_type &rgn, const point_type &delta, const region_type &limitRegion)
{
	return point_type(
		(delta.x < 0) ?
			(rgn.left <= limitRegion.left ? real_type(0) : (rgn.left + delta.x >= limitRegion.left ? delta.x : limitRegion.left - rgn.left)) :
			(rgn.right >= limitRegion.right ? real_type(0) : (rgn.right + delta.x <= limitRegion.right ? delta.x : limitRegion.right - rgn.right)),
		(delta.y < 0) ?
			(rgn.bottom <= limitRegion.bottom ? real_type(0) : (rgn.bottom + delta.y >= limitRegion.bottom ? delta.y : limitRegion.bottom - rgn.bottom)) :
			(rgn.top >= limitRegion.top ? real_type(0) : (rgn.top + delta.y <= limitRegion.top ? delta.y : limitRegion.top - rgn.top))
	);
}

/*static*/ RegionOfInterest::real_type RegionOfInterest::getSquareDistance(const point_type &pt1, const point_type &pt2)
{
	return (pt2.x-pt1.x)*(pt2.x-pt1.x) + (pt2.y-pt1.y)*(pt2.y-pt1.y);
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
LineROI::LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
LineROI::LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name), pt1_(pt1), pt2_(pt2)
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

RegionOfInterest * LineROI::clone() const
{
	return new LineROI(*this);
}

LineROI::points_type LineROI::points() const
{
	points_type points;
	points.push_back(pt1_);
	points.push_back(pt2_);
	return points;
}

bool LineROI::moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol)
{
	const real_type &dist1 = getSquareDistance(pt, pt1_);
	const real_type &dist2 = getSquareDistance(pt, pt2_);
	if (dist1 <= dist2)
	{
		if (dist1 < vertexTol*vertexTol)
		{
			pt1_ += delta;
			return true;
		}
		else return false;
	}
	else
	{
		if (dist2 < vertexTol*vertexTol)
		{
			pt2_ += delta;
			return true;
		}
		else return false;
	}
}

bool LineROI::moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol)
{
	const real_type &dist1 = getSquareDistance(pt, pt1_);
	const real_type &dist2 = getSquareDistance(pt, pt2_);
	if (dist1 <= dist2)
	{
		if (dist1 < vertexTol*vertexTol)
		{
			pt1_ += getMovableDistance(pt1_, delta, limitRegion);
			return true;
		}
		else return false;
	}
	else
	{
		if (dist2 < vertexTol*vertexTol)
		{
			pt2_ += getMovableDistance(pt2_, delta, limitRegion);
			return true;
		}
		else return false;
	}
}

void LineROI::moveRegion(const point_type &delta)
{
	pt1_ += delta;
	pt2_ += delta;
}

void LineROI::moveRegion(const point_type &delta, const region_type &limitRegion)
{
	// calculate the rectangular hull
	const region_type rgn(pt1_, pt2_);

	//
	const point_type &disp = getMovableDistance(rgn, delta, limitRegion);

	pt1_ += disp;
	pt2_ += disp;
}

bool LineROI::isVertex(const point_type &pt, const real_type &vertexTol) const
{
	return isNearPoint(pt1_, pt, vertexTol) || isNearPoint(pt2_, pt, vertexTol);
}

bool LineROI::include(const point_type &pt, const real_type &tol) const
{
	return LineSegment2<real_type>(pt1_, pt2_).include(pt, tol);
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
RectangleROI::RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
RectangleROI::RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name), rect_(pt1, pt2)
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

RegionOfInterest * RectangleROI::clone() const
{
	return new RectangleROI(*this);
}

RectangleROI::points_type RectangleROI::points() const
{
	points_type points;
	points.push_back(point_type(rect_.left, rect_.bottom));
	points.push_back(point_type(rect_.right, rect_.bottom));
	points.push_back(point_type(rect_.right, rect_.top));
	points.push_back(point_type(rect_.left, rect_.top));
	points.push_back(point_type(rect_.left, rect_.bottom));
	return points;
}

bool RectangleROI::moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol)
{
	const point_type pts[4] = {
		point_type(rect_.left, rect_.bottom),
		point_type(rect_.right, rect_.bottom),
		point_type(rect_.right, rect_.top),
		point_type(rect_.left, rect_.top)
	};

	std::vector<real_type> dists;
	dists.reserve(4);
	for (int i = 0; i < 4; ++i)
		dists.push_back(getSquareDistance(pts[i], pt));

	std::vector<real_type>::iterator it = std::min_element(dists.begin(), dists.end());
	if (*it < vertexTol*vertexTol)
	{
		const size_t idx = std::distance(dists.begin(), it);
		switch (idx)
		{
		case 0:
			rect_.left += delta.x;
			rect_.bottom += delta.y;
			return true;
		case 1:
			rect_.right += delta.x;
			rect_.bottom += delta.y;
			return true;
		case 2:
			rect_.right += delta.x;
			rect_.top += delta.y;
			return true;
		case 3:
			rect_.left += delta.x;
			rect_.top += delta.y;
			return true;
		default:
			return false;
		}
	}
	else return false;
}

bool RectangleROI::moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol)
{
	const point_type pts[4] = {
		point_type(rect_.left, rect_.bottom),
		point_type(rect_.right, rect_.bottom),
		point_type(rect_.right, rect_.top),
		point_type(rect_.left, rect_.top)
	};

	std::vector<real_type> dists;
	dists.reserve(4);
	for (int i = 0; i < 4; ++i)
		dists.push_back(getSquareDistance(pts[i], pt));

	std::vector<real_type>::iterator it = std::min_element(dists.begin(), dists.end());
	if (*it < vertexTol*vertexTol)
	{
		const size_t idx = std::distance(dists.begin(), it);
		const point_type &disp = getMovableDistance(pts[idx], delta, limitRegion);
		switch (idx)
		{
		case 0:
			rect_.left += disp.x;
			rect_.bottom += disp.y;
			return true;
		case 1:
			rect_.right += disp.x;
			rect_.bottom += disp.y;
			return true;
		case 2:
			rect_.right += disp.x;
			rect_.top += disp.y;
			return true;
		case 3:
			rect_.left += disp.x;
			rect_.top += disp.y;
			return true;
		default:
			return false;
		}
	}
	else return false;
}

void RectangleROI::moveRegion(const point_type &delta)
{
	rect_ += delta;
}

void RectangleROI::moveRegion(const point_type &delta, const region_type &limitRegion)
{
	rect_ += getMovableDistance(rect_, delta, limitRegion);
}

bool RectangleROI::isVertex(const point_type &pt, const real_type &vertexTol) const
{
	return isNearPoint(point_type(rect_.left, rect_.top), pt, vertexTol) || isNearPoint(point_type(rect_.left, rect_.bottom), pt, vertexTol) ||
		isNearPoint(point_type(rect_.right, rect_.top), pt, vertexTol) || isNearPoint(point_type(rect_.right, rect_.bottom), pt, vertexTol);
}

bool RectangleROI::include(const point_type &pt, const real_type &tol) const
{
	return rect_.left-tol <= pt.x && pt.x <= rect_.right+tol && rect_.bottom-tol <= pt.y && pt.y <= rect_.top+tol;
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
ROIWithVariablePoints::ROIWithVariablePoints(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
ROIWithVariablePoints::ROIWithVariablePoints(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name), points_()
{
}

#if defined(UNICODE) || defined(_UNICODE)
ROIWithVariablePoints::ROIWithVariablePoints(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
ROIWithVariablePoints::ROIWithVariablePoints(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name), points_(points)
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

ROIWithVariablePoints::point_type ROIWithVariablePoints::getPoint(const size_t index) const
{
	points_type::const_iterator it = points_.begin();
	std::advance(it, index);
	if (points_.end() == it)
		throw LogException(LogException::L_ERROR, "invalid index", __FILE__, __LINE__, __FUNCTION__);
	else return *it;
}

bool ROIWithVariablePoints::moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol)
{
	std::list<real_type> dists;
	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
		dists.push_back(getSquareDistance(*it, pt));

	std::list<real_type>::iterator it = std::min_element(dists.begin(), dists.end());
	if (*it < vertexTol*vertexTol)
	{
		const size_t idx = std::distance(dists.begin(), it);
		points_type::iterator itVertex = points_.begin();
		std::advance(itVertex, idx);
		*itVertex += delta;
		return true;
	}
	else return false;
}

bool ROIWithVariablePoints::moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol)
{
	std::list<real_type> dists;
	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
		dists.push_back(getSquareDistance(*it, pt));

	std::list<real_type>::iterator it = std::min_element(dists.begin(), dists.end());
	if (*it < vertexTol*vertexTol)
	{
		const size_t idx = std::distance(dists.begin(), it);
		points_type::iterator itVertex = points_.begin();
		std::advance(itVertex, idx);
		*itVertex += getMovableDistance(*itVertex, delta, limitRegion);
		return true;
	}
	else return false;
}

void ROIWithVariablePoints::moveRegion(const point_type &delta)
{
	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
		*it += delta;
}

void ROIWithVariablePoints::moveRegion(const point_type &delta, const region_type &limitRegion)
{
	if (points_.empty()) return;

	// calculate the rectangular hull
	points_type::iterator it = points_.begin();
	region_type rgn(*it, *it);
	++it;
	for (; it != points_.end(); ++it)
	{
		if (it->x < rgn.left) rgn.left = it->x;
		else if (it->x > rgn.right) rgn.right = it->x;
		if (it->y < rgn.bottom) rgn.bottom = it->y;
		else if (it->y > rgn.top) rgn.top = it->y;
	}

	//
	const point_type &disp = getMovableDistance(rgn, delta, limitRegion);
	for (points_type::iterator it = points_.begin(); it != points_.end(); ++it)
		*it += disp;
}

bool ROIWithVariablePoints::isVertex(const point_type &pt, const real_type &vertexTol) const
{
	for (points_type::const_iterator it = points_.begin(); it != points_.end(); ++it)
		if (isNearPoint(*it, pt, vertexTol)) return true;
	return false;
}

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
PolylineROI::PolylineROI(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
PolylineROI::PolylineROI(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name)
{
}

#if defined(UNICODE) || defined(_UNICODE)
PolylineROI::PolylineROI(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
PolylineROI::PolylineROI(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(points, isVisible, lineWidth, pointSize, lineColor, pointColor, name)
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

RegionOfInterest * PolylineROI::clone() const
{
	return new PolylineROI(*this);
}

PolylineROI::points_type PolylineROI::points() const
{
	return points_;
}

bool PolylineROI::include(const point_type &pt, const real_type &tol) const
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

//-----------------------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
PolygonROI::PolygonROI(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
PolygonROI::PolygonROI(const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(isVisible, lineWidth, pointSize, lineColor, pointColor, name)
{
}

#if defined(UNICODE) || defined(_UNICODE)
PolygonROI::PolygonROI(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::wstring &name /*= std::wstring()*/)
#else
PolygonROI::PolygonROI(const points_type &points, const bool isVisible, const real_type &lineWidth, const real_type &pointSize, const color_type &lineColor, const color_type &pointColor, const std::string &name /*= std::string()*/)
#endif
: base_type(points, isVisible, lineWidth, pointSize, lineColor, pointColor, name)
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

RegionOfInterest * PolygonROI::clone() const
{
	return new PolygonROI(*this);
}

PolygonROI::points_type PolygonROI::points() const
{
	if (points_.empty()) return points_;
	else
	{
		points_type pts(points_);
		pts.push_back(points_.front());
		return pts;
	}
}

bool PolygonROI::include(const point_type &pt, const real_type &tol) const
{
	if (points_.size() < 3) return false;
	return GeometryUtil::within(pt, points_, tol);
}

}  // namespace swl
