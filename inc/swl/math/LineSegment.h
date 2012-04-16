#if !defined(__SWL_MATH__LINE_SEGMENT__H_)
#define __SWL_MATH__LINE_SEGMENT__H_ 1


#include "swl/math/Vector.h"
#include "swl/math/Line.h"
#include "swl/math/Plane.h"
#include "swl/base/Point.h"
#include <limits>

namespace swl {

//-----------------------------------------------------------------------------------------
// class LineSegment2

template<typename T>
class LineSegment2
{
public:
    typedef T value_type;
    typedef Point2<T> point_type;
    typedef Vector2<T> vector_type;

public:
    explicit LineSegment2(const point_type &pt1 = point_type(), const point_type &pt2 = point_type())
	: point1_(pt1), point2_(pt2)
	{}
	explicit LineSegment2(const point_type rhs[2])
	: point1_(rhs[0]), point2_(rhs[1])
	{}
	LineSegment2(const LineSegment2<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_)
	{}
    ~LineSegment2()  {}

    LineSegment2<T> & operator=(const LineSegment2<T> &rhs)
    {
        if (this == &rhs) return *this;
        point1_ = rhs.point1_;
		point2_ = rhs.point2_;
        return *this;
    }

public:
	point_type & point1()  {  return point1_;  }
	const point_type & point1() const  {  return point1_;  }
	point_type & point2()  {  return point2_;  }
	const point_type & point2() const  {  return point2_;  }

	T getSlope(const T &tol = T(1.0e-5)) const
	{  return (point2_.x-point1_.x >= -tol && point2_.x-point1_.x <= tol) ? std::numeric_limits<T>::infinity() : (point2_.y-point1_.y) / (point2_.x-point1_.x);  }

	//
	bool isEqual(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{
		return (isEqual(point1_, line.point1_, tol) && isEqual(point2_, line.point2_, tol)) ||
			(isEqual(point1_, line.point2_, tol) && isEqual(point2_, line.point1_, tol));
	}
	bool isOrthogonal(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isOrthogonal(vector_type(line.point2_ - line.point1_), tol);  }
	bool isParallel(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isParallel(vector_type(line.point2_ - line.point1_), tol);  }
	bool isCollinear(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{  return toLine().isCollinear(line.toLine(), tol);  }
	bool isOverlapped(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{  return isCollinear(line, tol) && (include(line.point1_, tol) || include(line.point2_, tol) || line.include(point1_, tol) || line.include(point2_, tol));  }
	// don't consider the same or parallel, overlapped lines
	bool isIntersectedWith(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{
		const T x1 = point1_.x, y1 = point1_.y;
		const T x2 = line.point1_.x, y2 = line.point1_.y;
		const T a1 = point2_.x - point1_.x, b1 = point2_.y - point1_.y;
		const T a2 = line.point2_.x - line.point1_.x, b2 = line.point2_.y - line.point1_.y;

		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (y1 - y2) - b2 * (x1 - x2);
		const T num_s = a1 * (y1 - y2) - b1 * (x1 - x2);

		const bool val1 = denom >= -tol && denom <= tol;
		const bool val2 = num_t >= -tol && num_t <= tol;
		const bool val3 = num_s >= -tol && num_s <= tol;
#if 0
		if (val1 && val2 && val3)  // the same lines
			return include(line.point1_, tol) || include(line.point2_, tol) || line.include(point1_, tol) || line.include(point2_, tol);
		else if (val1) return false;  // parallel lines: meet at infinity
#else
		if (val1) return false;  // the same line or parallel lines: meet at infinity
#endif

		const T t = num_t / denom;
		const T s = num_s / denom;
		return t >= T(0) && t <= T(1) && s >= T(0) && s <= T(1);
	}

	// don't consider the same or parallel, overlapped lines
	point_type getIntersectionPoint(const LineSegment2<T> &line) const
	{
		const T x1 = point1_.x, y1 = point1_.y;
		const T x2 = line.point1_.x, y2 = line.point1_.y;
		const T a1 = point2_.x - point1_.x, b1 = point2_.y - point1_.y;
		const T a2 = line.point2_.x - line.point1_.x, b2 = line.point2_.y - line.point1_.y;

		const T tol = T(1.0e-5);
		if (isParallel(line, tol)) return point_type(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());

		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (y1 - y2) - b2 * (x1 - x2);
		const T num_s = a1 * (y1 - y2) - b1 * (x1 - x2);

		const T t = num_t / denom;
		const T s = num_s / denom;
		if (t >= T(0) && t <= T(1) && s >= T(0) && s <= T(1))
			return point_type(x1 + t * a1, y1 + t * b1);
		// TODO [check] >>
		else  // intersect outside of line segments
			return point_type(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());
	}

	//
	T getPerpendicularDistance(const point_type &pt) const
	{
		const vector_type dir(point2_ - point1_);
		const T norm = dir.norm();

		const T eps = T(1.0e-15);
		return (norm <= eps) ? vector_type(pt - point1_).norm() : (T)std::fabs(dir.y() * (pt.x - point1_.x) - dir.x() * (pt.y - point1_.y)) / norm;
	}
	point_type getPerpendicularPoint(const point_type &pt) const
	{
		const vector_type dir(point2_ - point1_);
		const T norm = dir.norm();

		const T eps = T(1.0e-15);
		if (norm <= eps) return point1_;
		else
		{
			const T s = (dir.x() * (pt.x - point1_.x) + dir.y() * (pt.y - point1_.y)) / norm;
			return point_type(dir.x() * s + point1_.x, dir.y() * s + point1_.y);
		}
	}
	point_type getClosestPoint(const point_type &pt) const
	{
		const T eps = T(1.0e-15);
		if (include(pt, eps)) return pt;
		else
		{
			const point_type &ppt = getPerpendicularPoint(pt);
			if (include(ppt, eps)) return ppt;
			else
			{
				const T dist1 = (point1_.x - pt.x)*(point1_.x - pt.x) + (point1_.y - pt.y)*(point1_.y - pt.y);
				const T dist2 = (point2_.x - pt.x)*(point2_.x - pt.x) + (point2_.y - pt.y)*(point2_.y - pt.y);

				return dist1 <= dist2 ? point1_ : point2_;
			}
		}
	}

	bool include(const point_type &pt, const T &tol) const
	{
#if 0
		return getPerpendicularDistance(pt) <= tol &&
			((std::min(point1_.x, point2_.x) <= pt.x && pt.x <= std::max(point1_.x, point2_.x)) ||
			(std::min(point1_.y, point2_.y) <= pt.y && pt.y <= std::max(point1_.y, point2_.y)));
#else
		return toLine().include(pt, tol) &&
			((std::min(point1_.x, point2_.x) <= pt.x && pt.x <= std::max(point1_.x, point2_.x)) ||
			(std::min(point1_.y, point2_.y) <= pt.y && pt.y <= std::max(point1_.y, point2_.y)));
#endif
	}

	Line2<T> toLine() const  {  return Line2<T>(point1_, point2_);  }

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol);  }

private:
	point_type point1_, point2_;
};


//-----------------------------------------------------------------------------------------
// class LineSegment3

template<typename T>
class LineSegment3
{
public:
    typedef T value_type;
    typedef Point3<T> point_type;
    typedef Vector3<T> vector_type;

public:
    explicit LineSegment3(const point_type &pt1 = point_type(), const point_type &pt2 = point_type())
	: point1_(pt1), point2_(pt2)
	{}
	explicit LineSegment3(const point_type rhs[2])
	: point1_(rhs[0]), point2_(rhs[1])
	{}
	LineSegment3(const LineSegment3<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_)
	{}
    ~LineSegment3()  {}

    LineSegment3<T> & operator=(const LineSegment3<T> &rhs)
    {
        if (this == &rhs) return *this;
        point1_ = rhs.point1_;
		point2_ = rhs.point2_;
        return *this;
    }

public:
	point_type & point1()  {  return point1_;  }
	const point_type & point1() const  {  return point1_;  }
	point_type & point2()  {  return point2_;  }
	const point_type & point2() const  {  return point2_;  }

	vector_type getDirectionalVector() const  {  return (point2_ - point1_).unit();  }

	//
	bool isEqual(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{
		return (isEqual(point1_, line.point1_, tol) && isEqual(point2_, line.point2_, tol)) ||
			(isEqual(point1_, line.point2_, tol) && isEqual(point2_, line.point1_, tol));
	}
	bool isOrthogonal(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isOrthogonal(vector_type(line.point2_ - line.point1_), tol);  }
	bool isParallel(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isParallel(vector_type(line.point2_ - line.point1_), tol);  }
	bool isCoplanar(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{
		const vector_type &normal = vector_type(point2_ - point1_).cross(vector_type(line.point1_ - point1_)).unit();
		const T delta = normal.x() * (line.point1_.x - point1_.x) + normal.y() * (line.point1_.y - point1_.y) + normal.z() * (line.point1_.z - point1_.z);
		return delta >= -tol && delta <= tol;
	}
	bool isCollinear(const LineSegment2<T> &line, const T &tol = T(1.0e-5)) const
	{  return toLine().isCollinear(line.toLine(), tol);  }
	bool isOverlapped(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{  return isCollinear(line, tol) && (include(line.point1_, tol) || include(line.point2_, tol) || line.include(point1_, tol) || line.include(point2_, tol));  }
	// don't consider the same or parallel, overlapped lines
	bool isIntersectedWith(const LineSegment3<T> &line, const T &tol = T(1.0e-5)) const
	{
		if (!isCoplanar(line, tol)) return false;
		else if (isOverlapped(line, tol)) return true;

		const T x1 = point1_.x, y1 = point1_.y, z1 = point1_.z;
		const T x2 = line.point1_.x, y2 = line.point1_.y, z2 = line.point1_.z;
		const T a1 = point2_.x - point1_.x, b1 = point2_.y - point1_.y, c1 = point2_.z - point1_.z;
		const T a2 = line.point2_.x - line.point1_.x, b2 = line.point2_.y - line.point1_.y, c2 = line.point2_.z - line.point1_.z;

		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (y1 - y2) - b2 * (x1 - x2);
		const T num_s = a1 * (y1 - y2) - b1 * (x1 - x2);

		const bool val1 = denom >= -tol && denom <= tol;
		const bool val2 = num_t >= -tol && num_t <= tol;
		const bool val3 = num_s >= -tol && num_s <= tol;
#if 0
		if (val1 && val2 && val3)  // the same lines
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
		else if (val1)  // parallel lines: meet at infinity
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
#else
		if (val1) return false;  // the same line or parallel lines: meet at infinity
#endif

		const T t = num_t / denom;
		const T s = num_s / denom;

		const T delta = c1 * t + z1 - (c2 * s + z2);
		if (delta < -tol && delta > tol)  // a case that lines don't intersect
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		return t >= T(0) && t <= T(1) && s >= T(0) && s <= T(1);
	}

	// don't consider the same or parallel, overlapped lines
	point_type getIntersectionPoint(const LineSegment3<T> &line) const
	{
		const T tol = T(1.0e-5);

		if (!isCoplanar(line, tol) || isParallel(line, tol))  // don't intersect with each other
			return point_type(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

		const T x1 = point1_.x, y1 = point1_.y, z1 = point1_.z;
		const T x2 = line.point1_.x, y2 = line.point1_.y, z2 = line.point1_.z;
		const T a1 = point2_.x - point1_.x, b1 = point2_.y - point1_.y, c1 = point2_.z - point1_.z;
		const T a2 = line.point2_.x - line.point1_.x, b2 = line.point2_.y - line.point1_.y, c2 = line.point2_.z - line.point1_.z;

		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (y1 - y2) - b2 * (x1 - x2);
		const T num_s = a1 * (y1 - y2) - b1 * (x1 - x2);

		const bool val1 = denom >= -tol && denom <= tol;
		const bool val2 = num_t >= -tol && num_t <= tol;
		const bool val3 = num_s >= -tol && num_s <= tol;
#if 0
		if (val1 && val2 && val3)  // the same lines
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
		else if (val1)  // parallel lines: meet at infinity
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
#else
		if (val1)  // the same line or parallel lines: meet at infinity
			return point_type(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
#endif

		const T t = num_t / denom;
		const T s = num_s / denom;

		if (t >= T(0) && t <= T(1) && s >= T(0) && s <= T(1))
		{
			const T delta = c1 * t + z1 - (c2 * s + z2);
			if (delta < -tol && delta > tol)  // a case that lines don't intersect
				throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

			return point_type(a1 * t + x1, b1 * t + y1, c1 * t + z1);
		}
		// TODO [check] >>
		else  // intersect outside of line segments
			return point_type(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());
	}

	LineSegment3<T> getCommonNormal(const LineSegment3<T> &line) const
	{
		const vector_type &dir1 = getDirectionalVector();
		const vector_type &dir2 = line.getDirectionalVector();
		// TODO [check] >>
		if (dir1.isEqual(dir2))  // infinitely many common normals exist on planes with dir1 or dir2 as a normal vector
			return LineSegment3<T>();
		else if (dir1.isParallel(dir2))  // infinitely many common normals exist on a plane defined by two lines
			return LineSegment3<T>();

		const T eps = T(1.0e-15);
		const vector_type &normal = dir1.cross(dir2);
		if (normal.norm() <= eps)  // error
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		const T &a1 = dir1.x, &b1 = dir1.y, &c1 = dir1.z;
		const T &a2 = dir2.x, &b2 = dir2.y, &c2 = dir2.z;
		const T &nx = normal.x(), &ny = normal.y(), &nz = normal.z();
		const T &x1 = point1_.x, &y1 = point1_.y, &z1 = point1_.z;
		const T &x2 = line.point1_.x, &y2 = line.point1_.y, &z2 = line.point1_.z;

		const T det0 = (T)MathExt::det(a1, -a2, nx, b1, -b2, ny, c1, -c2, nz);
		if (det0 >= -eps && det0 <= eps)  // error
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
		const T det1 = (T)MathExt::det(x2 - x1, -a2, nx, y2 - y1, -b2, ny, z2 - z1, -c2, nz);
		const T det2 = (T)MathExt::det(a1, x2 - x1, nx, b1, y2 - y1, ny, c1, z2 - z1, nz);
		//const T det3 = (T)MathExt::det(a1, -a2, x2 - x1, b1, -b2, y2 - y1, c1, -c2, z2 - z1);

		const T t = det1 / det0;
		const T s = det2 / det0;
		//const T r = det3 / det0;

		return LineSegment3<T>(point_type(x1 + a1 * t, y1 + b1 * t, z1 + c1 * t), point_type(x2 + a2 * s, y2 + b2 * s, z2 + c2 * s));
	}

	//
	T getPerpendicularDistance(const point_type &pt) const
	{
		const point_type &ppt = getPerpendicularPoint(pt);
		return vector_type(ppt - pt).norm();
	}
	point_type getPerpendicularPoint(const point_type &pt) const
	{
		const vector_type dir(point2_ - point1_);
		const T norm = dir.norm();

		const T eps = T(1.0e-15);
		if (norm <= eps) return point1_;
		else
		{
			const T s = (dir.x() * (pt.x - point1_.x) + dir.y() * (pt.y - point1_.y) + dir.z() * (pt.z - point1_.z)) / norm;
			return point_type(dir.x() * s + point1_.x, dir.y() * s + point1_.y, dir.z() * s + point1_.z);
		}
	}
	point_type getClosestPoint(const point_type &pt) const
	{
		const T eps = T(1.0e-15);
		if (include(pt, eps)) return pt;
		else
		{
			const point_type &ppt = getPerpendicularPoint(pt);
			if (include(ppt, eps)) return ppt;
			else
			{
				const T dist1 = (point1_.x - pt.x)*(point1_.x - pt.x) + (point1_.y - pt.y)*(point1_.y - pt.y) + (point1_.z - pt.z)*(point1_.z - pt.z);
				const T dist2 = (point2_.x - pt.x)*(point2_.x - pt.x) + (point2_.y - pt.y)*(point2_.y - pt.y) + (point2_.z - pt.z)*(point2_.z - pt.z);

				return dist1 <= dist2 ? point1_ : point2_;
			}
		}
	}

	bool include(const point_type &pt, const T &tol) const
	{
#if 0
		return getPerpendicularDistance(pt) <= tol &&
			((std::min(point1_.x, point2_.x) <= pt.x && pt.x <= std::max(point1_.x, point2_.x)) ||
			(std::min(point1_.y, point2_.y) <= pt.y && pt.y <= std::max(point1_.y, point2_.y)) ||
			(std::min(point1_.z, point2_.z) <= pt.z && pt.z <= std::max(point1_.z, point2_.z)));
#else
		return toLine().include(pt, tol) &&
			((std::min(point1_.x, point2_.x) <= pt.x && pt.x <= std::max(point1_.x, point2_.x)) ||
			(std::min(point1_.y, point2_.y) <= pt.y && pt.y <= std::max(point1_.y, point2_.y)) ||
			(std::min(point1_.z, point2_.z) <= pt.z && pt.z <= std::max(point1_.z, point2_.z)));
#endif
	}

	Line3<T> toLine() const  {  return Line3<T>(point1_, point2_);  }

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol) && (pt1.z-pt2.z >= -tol && pt1.z-pt2.z <= tol);  }

private:
	point_type point1_, point2_;
};

}  // namespace swl


#endif  //  __SWL_MATH__LINE_SEGMENT__H_
