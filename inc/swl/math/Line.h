#if !defined(__SWL_MATH__LINE__H_)
#define __SWL_MATH__LINE__H_ 1


#include "swl/math/Vector.h"
#include "swl/math/Plane.h"
#include "swl/math/MathExt.h"
#include "swl/base/Point.h"
#include <limits>

namespace swl {

//-----------------------------------------------------------------------------------------
// class Line2

template<typename T>
class Line2
{
public:
    typedef T value_type;
    typedef Point2<T> point_type;
    typedef Vector2<T> vector_type;

public:
    explicit Line2(const point_type &pt1 = point_type(), const point_type &pt2 = point_type())
	: point1_(pt1), point2_(pt2)
	{}
	explicit Line2(const point_type rhs[2])
	: point1_(rhs[0]), point2_(rhs[1])
	{}
    explicit Line2(const Line2<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_)
	{}
    ~Line2()  {}

    Line2<T> & operator=(const Line2<T> &rhs)
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

	T getSlope() const
	{  return (point2_.x-point1_.x >= -tol && point2_.x-point1_.x <= tol) ? std::numeric_limits<T>::infinity() : ((point2_.y-point1_.y) / (point2_.x-point1_.x));  }
	T getIntercept() const
	{  return (point2_.x-point1_.x >= -tol && point2_.x-point1_.x <= tol) ? std::numeric_limits<T>::infinity() : (point1_.y - point1_.x * (point2_.y-point1_.y) / (point2_.x-point1_.x));  }

	//
	bool isEqual(const Line2<T> &line, const T &tol = T(1.0e-5)) const
	{  return isParallel(line) && (contain(line.point1_, tol) || contain(line.point2_, tol));  }
	bool isOrthogonal(const Line2<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isOrthogonal(vector_type(line.point2_ - line.point1_));  }
	bool isParallel(const Line2<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isParallel(vector_type((line.point2_ - line.point1_));  }

	bool isIntersectedWith(const Line2<T> &line, const T &tol = T(1.0e-5)) const
	{  return isEqual(line, tol) || !isParallel(line, tol);  }
	point_type getIntersectionPoint(const Line2<T> &line) const
	{
		const T a1 = point2_.x - point1_.x, b1 = point2_.y - point1_.y;
		const T a2 = line.point2_.x - line.point1_.x, b2 = line.point2_.y - line.point1_.y;

#if 0
		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (point1_.y - line.point1_.y) - b2 * (point1_.x - line.point1_.x);
		const T num_s = a1 * (point1_.y - line.point1_.y) - b1 * (point1_.x - line.point1_.x);

		const bool val1 = denom >= -tol && denom <= tol;
		const bool val2 = num_t >= -tol && num_t <= tol;
		const bool val3 = num_s >= -tol && num_s <= tol;
		if (val1 && val2 && val3)  // the same lines
		{
			// TODO [implement] >>
		}
		else if (val1)  // parallel lines: meet at infinity
			return point_type(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());
#else
		const T tol = T(1.0e-5);

		// don't consider the same or parallel lines
		if (isParallel(line, tol)) return point_type(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());

		const T denom = b2 * a1 - a2 * b1;
		const T num_t = a2 * (point1_.y - line.point1_.y) - b2 * (point1_.x - line.point1_.x);
		const T num_s = a1 * (point1_.y - line.point1_.y) - b1 * (point1_.x - line.point1_.x);
#endif

		const T t = num_t / denom;
		//const float s = num_s / denom;
		return point_type(point1_.x + t * (point2_.x - point1_.x), point1_.y + t * (b1));
	}

	//
	T getPerpendicularDistance(const point_type &pt) const
	{
		const vector_type dir(point2_ - point1_);
		const T norm = dir.norm();

		const T eps= T(1.0e-15);
		return (norm <= eps) ? vector_type(pt - point1_).norm() : (T)std::fabs(dir.y * (pt.x - point1_.x) - dir.x * (pt.y - point1_.y)) / norm;
	}
	point_type getPerpendicularPoint(const point_type &pt) const
	{
		const vector_type dir(point2_ - point1_);
		const T norm = dir.norm();

		const T eps= T(1.0e-15);
		if (norm <= eps) return point1_;
		else
		{
			const T s = (dir.x * (pt.x - point1_.x) + dir.y * (pt.y - point1_.y)) / norm;
			return point_type(dir.x * s + point1_.x, dir.y * s + point1_.y);
		}
	}

	bool contain(const point_type &pt, const T &tol) const
	{  return getPerpendicularDistance(pt) <= tol;  }

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol);  }

public:
	point_type point1_, point2_;
};


//-----------------------------------------------------------------------------------------
// class Line3

template<typename T>
struct Line3
{
public:
    typedef T value_type;
    typedef Point3<T> point_type;
    typedef Vector3<T> vector_type;

public:
    explicit Line3(const point_type &pt1 = point_type(), const point_type &pt2 = point_type())
	: point1_(pt1), point2_(pt2)
	{}
	explicit Line3(const point_type rhs[2])
	: point1_(rhs[0]), point2_(rhs[1])
	{}
    explicit Line3(const Line3<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_)
	{}
    ~Line3()  {}

    Line3<T> & operator=(const Line3<T> &rhs)
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
	bool isEqual(const Line3<T> &line, const T &tol = T(1.0e-5)) const
	{  return isParallel(line) && (contain(line.point1_, tol) || contain(line.point2_, tol));  }
	bool isOrthogonal(const Line3<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isOrthogonal(vector_type((line.point2_ - line.point1_),tol);  }
	bool isParallel(const Line3<T> &line, const T &tol = T(1.0e-5)) const
	{  return vector_type(point2_ - point1_).isParallel(vector_type((line.point2_ - line.point1_), tol);  }

	bool isIntersectedWith(const Line3<T> &line, const T &tol = T(1.0e-5)) const
	{  return isEqual(line, tol) ? true : (isParallel(line, tol) ? false : isOnTheSamePlaneWith(line, tol));  }
	bool isOnTheSamePlaneWith(const Line3<T> &line, const T &tol = T(1.0e-5)) const
	{
		const Plane3<T> plane(point1_, point2_, line.point1_);
		return plane.contain(line.point2_, tol);
	}

	point_type getIntersectionPoint(const Line3<T> &line) const
	{
		const T tol = T(1.0e-5);

		// don't consider the same or parallel line
		if (!isOnTheSamePlaneWith(line, tol) || isParallel(line, tol))  // don't intersect with each other
			return point_type(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());

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
		if (val1 && val2 && val3)  // the same lines
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
		else if (val1) return false;  // parallel lines: meet at infinity
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		const T t = num_t / denom;
		const T s = num_s / denom;

		const T delta = c1 * t + z1 - (c2 * s + z2);
		if (delta < -tol && delta > tol)  // a case that lines don't intersect
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		return point_type(a1 * t + x1, b1 * t + y1, c1 * t + z1);
	}

	Line3<T> getCommonNormal(const Line3<T> &line) const
	{
		const vector_type &dir1 = getDirectionalVector();
		const vector_type &dir2 = line.getDirectionalVector();
		// TODO [check] >>
		if (dir1.isEqual(dir2))  // infinitely many common normals exist on planes with dir1 or dir2 as a normal vector
			return Line3<T>();
		else if (dir1.isParallel(dir2))  // infinitely many common normals exist on a plane defined by two lines
			return Line3<T>();

		const T eps = T(1.0e-15);
		const vector_type &normal = dir1.cross(dir2);
		if (normal.norm() <= eps)  // error
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		const T &a1 = dir1.x, &b1 = dir1.y, &c1 = dir1.z;
		const T &a2 = dir2.x, &b2 = dir2.y, &c2 = dir2.z;
		const T &nx = normal.x, &ny = normal.y, &nz = normal.z;
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

		return Line3<T>(point_type(x1 + a1 * t, y1 + b1 * t, z1 + c1 * t), point_type(x2 + a2 * s, y2 + b2 * s, z2 + c2 * s));
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

		const T eps= T(1.0e-15);
		if (norm <= eps) return point1_;
		else
		{
			const T s = (dir.x * (pt.x - point1_.x) + dir.y * (pt.y - point1_.y) + dir.z * (pt.z - point1_.z)) / norm;
			return point_type(dir.x * s + point1_.x, dir.y * s + point1_.y, dir.z * s + point1_.z);
		}
	}

	bool contain(const point_type &pt, const T &tol) const
	{  return getPerpendicularDistance(pt) <= tol;  }

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol) && (pt1.z-pt2.z >= -tol && pt1.z-pt2.z <= tol);  }

private:
	point_type point1_, point2_;
};

}  // namespace swl


#endif  //  __SWL_MATH__LINE__H_
