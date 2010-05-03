#if !defined(__SWL_UTIL__REGION_OF_INTEREST__H_)
#define __SWL_UTIL__REGION_OF_INTEREST__H_ 1


#include "swl/util/ExportUtil.h"
#include "swl/graphics/Color.h"
#include "swl/base/Region.h"
#include <list>
#include <string>


namespace swl {

//-----------------------------------------------------------------------------------------
//

class SWL_UTIL_API RegionOfInterest
{
public:
	//typedef RegionOfInterest base_type;
	typedef float real_type;
	typedef Point2<real_type> point_type;
	typedef std::list<point_type> points_type;
	typedef Region2<real_type> region_type;
	typedef Color4<real_type> color_type;

protected:
#if defined(UNICODE) || defined(_UNICODE)
	explicit RegionOfInterest(const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit RegionOfInterest(const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit RegionOfInterest(const RegionOfInterest &rhs);
public:
	virtual ~RegionOfInterest();

	RegionOfInterest & operator=(const RegionOfInterest &rhs);

public:
	virtual RegionOfInterest * clone() const = 0;

	virtual points_type points() const = 0;

	//
	virtual bool moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol) = 0;
	virtual bool moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol) = 0;
	virtual void moveRegion(const point_type &delta) = 0;
	virtual void moveRegion(const point_type &delta, const region_type &limitRegion) = 0;

	virtual bool isVertex(const point_type &pt, const real_type &radius) const = 0;
	virtual bool include(const point_type &pt, const real_type &tol) const = 0;

	//
	void setVisible(const bool isVisible)  {  isVisible_ = isVisible;  }
	bool isVisible() const  {  return isVisible_; } 

	void setColor(const color_type &color)  {  color_ = color;  }
	const color_type & getColor() const  {  return color_;  }

#if defined(UNICODE) || defined(_UNICODE)
	void setName(const std::wstring &name)  {  name_ = name;  }
	std::wstring getName() const  {  return name_;  }
#else
	void setName(const std::string &name)  {  name_ = name;  }
	std::string getName() const  {  return name_;  }
#endif

protected:
	static bool isNearPoint(const point_type &pt1, const point_type &pt2, const real_type &tol);
	static point_type getMovableDistance(const point_type &pt, const point_type &delta, const region_type &limitRegion);
	static point_type getMovableDistance(const region_type &rgn, const point_type &delta, const region_type &limitRegion);

	static real_type getSquareDistance(const point_type &pt1, const point_type &pt2);

protected:
	struct PrComparePoints
	{
		PrComparePoints(const point_type &point)
		: point_(point)
		{}

		bool operator()(const point_type &rhs) const;

	private:
		const point_type &point_;
	};

protected:
	bool isVisible_;
	color_type color_;
#if defined(UNICODE) || defined(_UNICODE)
	std::wstring name_;
#else
	std::string name_;
#endif
};

//-----------------------------------------------------------------------------------------
//

struct SWL_UTIL_API LineROI: public RegionOfInterest
{
public:
	typedef RegionOfInterest base_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	explicit LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit LineROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit LineROI(const LineROI &rhs);
	virtual ~LineROI();

	LineROI & operator=(const LineROI &rhs);

public:
	point_type point1() const  {  return pt1_;  }
	point_type point2() const  {  return pt2_;  }

	//
	/*virtual*/ RegionOfInterest * clone() const;

	/*virtual*/ points_type points() const;

	//
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol);
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol);
	/*virtual*/ void moveRegion(const point_type &delta);
	/*virtual*/ void moveRegion(const point_type &delta, const region_type &limitRegion);

	/*virtual*/ bool isVertex(const point_type &pt, const real_type &radius) const;
	/*virtual*/ bool include(const point_type &pt, const real_type &tol) const;

private:
	point_type pt1_, pt2_;
};

//-----------------------------------------------------------------------------------------
//

struct SWL_UTIL_API RectangleROI: public RegionOfInterest
{
public:
	typedef RegionOfInterest base_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	explicit RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit RectangleROI(const point_type &pt1, const point_type &pt2, const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit RectangleROI(const RectangleROI &rhs);
	virtual ~RectangleROI();

	RectangleROI & operator=(const RectangleROI &rhs);

public:
	point_type point1() const  {  return point_type(rect_.left, rect_.bottom);  }
	point_type point2() const  {  return point_type(rect_.right, rect_.top);  }

	//
	/*virtual*/ RegionOfInterest * clone() const;

	/*virtual*/ points_type points() const;

	//
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol);
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol);
	/*virtual*/ void moveRegion(const point_type &delta);
	/*virtual*/ void moveRegion(const point_type &delta, const region_type &limitRegion);

	/*virtual*/ bool isVertex(const point_type &pt, const real_type &radius) const;
	/*virtual*/ bool include(const point_type &pt, const real_type &tol) const;

private:
	region_type rect_;
};

//-----------------------------------------------------------------------------------------
//

class SWL_UTIL_API ROIWithVariablePoints: public RegionOfInterest
{
public:
	typedef RegionOfInterest base_type;

protected:
#if defined(UNICODE) || defined(_UNICODE)
	explicit ROIWithVariablePoints(const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
	explicit ROIWithVariablePoints(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit ROIWithVariablePoints(const bool isVisible, const color_type &color, const std::string &name = std::string());
	explicit ROIWithVariablePoints(const points_type &points, const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit ROIWithVariablePoints(const ROIWithVariablePoints &rhs);
public:
	virtual ~ROIWithVariablePoints();

	ROIWithVariablePoints & operator=(const ROIWithVariablePoints &rhs);

public:
	void addPoint(const point_type &point);
	void removePoint(const point_type &point);
	void clearAllPoints()  {  points_.clear();  }
	size_t countPoint() const  {  return points_.size();  }
	bool containPoint() const  {  return !points_.empty();  }

	point_type getPoint(const size_t index) const;

	//
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const real_type &vertexTol);
	/*virtual*/ bool moveVertex(const point_type &pt, const point_type &delta, const region_type &limitRegion, const real_type &vertexTol);
	/*virtual*/ void moveRegion(const point_type &delta);
	/*virtual*/ void moveRegion(const point_type &delta, const region_type &limitRegion);

	/*virtual*/ bool isVertex(const point_type &pt, const real_type &radius) const;

protected:
	points_type points_;
};

//-----------------------------------------------------------------------------------------
//

struct SWL_UTIL_API PolylineROI: public ROIWithVariablePoints
{
public:
	typedef ROIWithVariablePoints base_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	explicit PolylineROI(const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
	explicit PolylineROI(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit PolylineROI(const bool isVisible, const color_type &color, const std::string &name = std::string());
	explicit PolylineROI(const points_type &points, const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit PolylineROI(const PolylineROI &rhs);
	virtual ~PolylineROI();

	PolylineROI & operator=(const PolylineROI &rhs);

public:
	/*virtual*/ RegionOfInterest * clone() const;

	/*virtual*/ points_type points() const;

	//
	/*virtual*/ bool include(const point_type &pt, const real_type &tol) const;
};

//-----------------------------------------------------------------------------------------
//

struct SWL_UTIL_API PolygonROI: public ROIWithVariablePoints
{
public:
	typedef ROIWithVariablePoints base_type;

public:
#if defined(UNICODE) || defined(_UNICODE)
	explicit PolygonROI(const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
	explicit PolygonROI(const points_type &points, const bool isVisible, const color_type &color, const std::wstring &name = std::wstring());
#else
	explicit PolygonROI(const bool isVisible, const color_type &color, const std::string &name = std::string());
	explicit PolygonROI(const points_type &points, const bool isVisible, const color_type &color, const std::string &name = std::string());
#endif
	explicit PolygonROI(const PolygonROI &rhs);
	virtual ~PolygonROI();

	PolygonROI & operator=(const PolygonROI &rhs);

public:
	/*virtual*/ RegionOfInterest * clone() const;

	/*virtual*/ points_type points() const;

	//
	/*virtual*/ bool include(const point_type &pt, const real_type &tol) const;
};

}  // namespace swl


#endif  // __SWL_UTIL__REGION_OF_INTEREST__H_
