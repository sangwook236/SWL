#if !defined(__SWL_MATH__PLANE__H_)
#define __SWL_MATH__PLANE__H_ 1


#include "swl/math/Line.h"
#include "swl/math/Vector.h"
#include "swl/base/Point.h"

namespace swl {

//-----------------------------------------------------------------------------------------
// class Plane3

template<typename T>
struct Plane3
{
public:
    typedef T value_type;
    typedef Point3<T> point_type;
    typedef Vector3<T> vector_type;
    typedef Line3<T> line_type;

public:
    explicit Plane3(const point_type &pt1 = point_type(), const point_type &pt2 = point_type(), const point_type &pt3 = point_type())
	: point1_(pt1), point2_(pt2), point3_(pt3)
	{}
	explicit Plane3(const point_type rhs[3])
	: point1_(rhs[0]), point2_(rhs[1]), point3_(rhs[2])
	{}
    explicit Plane3(const Plane3<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_), point3_(rhs.point3_)
	{}
    ~Plane3()  {}

    Plane3<T> & operator=(const Plane3<T> &rhs)
    {
        if (this == &rhs) return *this;
        point1_ = rhs.point1_;
		point2_ = rhs.point2_;
		point3_ = rhs.point3_;
        return *this;
    }

public:    
	point_type & point1()  {  return point1_;  }
	const point_type & point1() const  {  return point1_;  }
	point_type & point2()  {  return point2_;  }
	const point_type & point2() const  {  return point2_;  }
	point_type & point3()  {  return point3_;  }
	const point_type & point3() const  {  return point3_;  }

	vector_type getNormalVector() const
	{  return vector_type(point2_ - point1_).cross(vector_type(point3_ - point1_)).unit();  }
	// a * x + b * y + c * z + d = 0
	void getPlaneEquation(T &a, T &b, T &c, T &d) const
	{
		const vector_type &normal = getNormalVector();
		a = normal.x;
		b = normal.y;
		c = normal.z;
		d = -a * point1_.x - b * point1_.y - c * point1_.z);
	}

	//
	bool isEqual(const Plane3<T> &plane, const T &tol = T(1.0e-5)) const
	{  return isParallel(plane) && (contain(plane.point1_, tol) || contain(plane.point2_, tol) || contain(plane.point3_, tol));  }
	bool isOrthogonal(const Plane3<T> &plane, const T &tol = T(1.0e-5)) const
	{  return getNormalVector().isOrthogonal(plane.getNormalVector());  }
	bool isParallel(const Plane3<T> &plane, const T &tol = T(1.0e-5)) const
	{  return getNormalVector().isParallel(plane.getNormalVector());  }

	bool isIntersectedWith(const Plane3<T> &plane, const T &tol = T(1.0e-5)) const
	{  return isEqual(line, tol) || !isParallel(line, tol);  }

	//
	point_type getIntersectionPoint(const line_type &line) const
	{
		const vector_type &normal = getNormalVector();
		const vector_type &dir = line.getDirectionalVector();

		if (normal.isOrthogonal(dir))  // meet at infinity
			return point_type(std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity());

		const T denom = normal.x * dir.x + normal.y * dir.y + normal.z * dir.z;
		if (denom >= -eps && denom <= eps)  // error
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

		const point_type &pt = line.point1();
		const T t = (-normal.x * pt.x - normal.y * pt.y - normal.z * pt.z) / denom;

		return point_type(dir.x * t + pt.x, dir.y * t + pt.y, dir.z * t + pt.z);
	}

	//
	T getPerpendicularDistance(const point_type &pt) const
	{
		const vector_type &normal = getNormalVector();
		const T norm = normal.norm();

		const T eps= T(1.0e-15);
		return (norm <= eps) ? vector_type(pt - point1_).norm() : (T)std::fabs(dir.y * (pt.x - point1_.x) - dir.x * (pt.y - point1_.y)) / norm;
	}
	point_type getPerpendicularPoint(const point_type &pt) const
	{
		const vector_type &normal = getNormalVector();
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
	point_type point1_, point2_, point3_;
};

}  // namespace swl


#endif  //  __SWL_MATH__PLANE__H_
