#if !defined(__SWL_MATH__TRIANGLE__H_)
#define __SWL_MATH__TRIANGLE__H_ 1


#include "swl/math/Vector.h"
#include "swl/math/Plane.h"

namespace swl {

//-----------------------------------------------------------------------------------------
// class Triangle2

template<typename T>
class Triangle2
{
public:
    typedef T value_type;
    typedef Point2<T> point_type;
    typedef Vector2<T> vector_type;

public:
    explicit Triangle2(const point_type &pt1 = point_type(), const point_type &pt2 = point_type(), const point_type &pt3 = point_type())
	: point1_(pt1), point2_(pt2), point3_(pt3)
	{}
	explicit Triangle2(const point_type rhs[3])
	: point1_(rhs[0]), point2_(rhs[1]), point3_(rhs[2])
	{}
    explicit Triangle2(const Triangle2<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_), point3_(rhs.point3_)
	{}
    ~Triangle2()  {}

    Triangle2<T> & operator=(const Triangle2<T> &rhs)
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

	//
	bool isEqual(const Triangle2<T> &tri, const T &tol = T(1.0e-5)) const
	{
		return (isEqual(point1_, tri.point1_, tol) && isEqual(point2_, tri.point2_, tol) && isEqual(point3_, tri.point3_, tol)) ||
			(isEqual(point1_, tri.point2_, tol) && isEqual(point2_, tri.point3_, tol) && isEqual(point3_, tri.point1_, tol)) ||
			(isEqual(point1_, tri.point3_, tol) && isEqual(point2_, tri.point1_, tol) && isEqual(point3_, tri.point2_, tol)) ||
			(isEqual(point1_, tri.point1_, tol) && isEqual(point2_, tri.point3_, tol) && isEqual(point3_, tri.point2_, tol)) ||
			(isEqual(point1_, tri.point3_, tol) && isEqual(point2_, tri.point2_, tol) && isEqual(point3_, tri.point1_, tol)) ||
			(isEqual(point1_, tri.point2_, tol) && isEqual(point2_, tri.point1_, tol) && isEqual(point3_, tri.point3_, tol));
	}
	bool isOverlapped(const Triangle2<T> &tri, const T &tol = T(1.0e-5)) const
	{
		return contain(tri.point1_, tol) || contain(tri.point2_, tol) || contain(tri.point3_, tol) ||
			tri.contain(point1_, tol) || tri.contain(point2_, tol) || tri.contain(point3_, tol);
	}

	bool contain(const point_type &pt, const T &tol) const
	{
		const point_type cx = (point1_.x + point2_.x + point3_.x) / T(3), cy = (point1_.y + point2_.y + point3_.y) / T(3);

		return isInside(pt.x, pt.y, cx, cy, point2_.x - point1_.x, point2_.y - point1_.y, point1_.x, point1_.y, tol) &&
			isInside(pt.x, pt.y, cx, cy, point3_.x - point2_.x, point3_.y - point2_.y, point2_.x, point2_.y, tol) &&
			isInside(pt.x, pt.y, cx, cy, point1_.x - point3_.x, point1_.y - point3_.y, point3_.x, point3_.y, tol);
	}

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol);  }

	// line equation: (x - x0) / a = (y - y0) / b ==> b * (x - x0) - a * (y - y0) = 0
	bool isInside(const T &px, const T &py, const T &cx, const T &cy, const T &a, const T &b, const T &x0, const T &y0, const T &tol) const
	{  return (b * (px - x0) - a * (py - y0)) * (b * (cx - x0) - a * (cy - y0)) >= T(0);  }

private:
	point_type point1_, point2_, point3_;
};


//-----------------------------------------------------------------------------------------
// class Triangle3

template<typename T>
struct Triangle3
{
public:
    typedef T value_type;
    typedef Point3<T> point_type;
    typedef Vector3<T> vector_type;

public:
    explicit Triangle3(const point_type &pt1 = point_type(), const point_type &pt2 = point_type(), const point_type &pt3 = point_type())
	: point1_(pt1), point2_(pt2), point3_(pt3)
	{}
	explicit Triangle3(const point_type rhs[3])
	: point1_(rhs[0]), point2_(rhs[1]), point3_(rhs[2])
	{}
    explicit Triangle3(const Triangle3<T> &rhs)
	: point1_(rhs.point1_), point2_(rhs.point2_), point3_(rhs.point3_)
	{}
    ~Triangle3()  {}

    Triangle3<T> & operator=(const Triangle3<T> &rhs)
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

	//
	bool isEqual(const Triangle3<T> &tri, const T &tol = T(1.0e-5)) const
	{
		return (isEqual(point1_, tri.point1_, tol) && isEqual(point2_, tri.point2_, tol) && isEqual(point3_, tri.point3_, tol)) ||
			(isEqual(point1_, tri.point2_, tol) && isEqual(point2_, tri.point3_, tol) && isEqual(point3_, tri.point1_, tol)) ||
			(isEqual(point1_, tri.point3_, tol) && isEqual(point2_, tri.point1_, tol) && isEqual(point3_, tri.point2_, tol)) ||
			(isEqual(point1_, tri.point1_, tol) && isEqual(point2_, tri.point3_, tol) && isEqual(point3_, tri.point2_, tol)) ||
			(isEqual(point1_, tri.point3_, tol) && isEqual(point2_, tri.point2_, tol) && isEqual(point3_, tri.point1_, tol)) ||
			(isEqual(point1_, tri.point2_, tol) && isEqual(point2_, tri.point1_, tol) && isEqual(point3_, tri.point3_, tol));
	}
	bool isOrthogonal(const Triangle3<T> &tri, const T &tol = T(1.0e-5)) const
	{  return getNormalVector().isOrthogonal(tri.getNormalVector());  }
	bool isParallel(const Triangle3<T> &tri, const T &tol = T(1.0e-5)) const
	{  return getNormalVector().isParallel(tri.getNormalVector());  }
	bool isCoplanar(const Triangle3<T> &tri, const T &tol = T(1.0e-5)) const
	{  return toPlane().isCollinear(tri.toPlane(), tol);  }
	bool isOverlapped(const Triangle3<T> &tri, const T &tol = T(1.0e-5)) const
	{
		return isCoplanar(tri, tol) &&
			(contain(tri.point1_, tol) || contain(tri.point2_, tol) || contain(tri.point3_, tol) ||
			tri.contain(point1_, tol) || tri.contain(point2_, tol) || tri.contain(point3_, tol));
	}

	//
	T getPerpendicularDistance(const point_type &pt) const
	{
		const vector_type &normal = getNormalVector();
		const T norm = normal.norm();

		const T eps = T(1.0e-15);
		return (norm <= eps) ? vector_type(pt - point1_).norm() : (T)std::fabs(dir.y * (pt.x - point1_.x) - dir.x * (pt.y - point1_.y)) / norm;
	}
	point_type getPerpendicularPoint(const point_type &pt) const
	{
		const vector_type &normal = getNormalVector();
		const T norm = dir.norm();

		const T eps = T(1.0e-15);
		if (norm <= eps) return point1_;
		else
		{
			const T s = (dir.x * (pt.x - point1_.x) + dir.y * (pt.y - point1_.y) + dir.z * (pt.z - point1_.z)) / norm;
			return point_type(dir.x * s + point1_.x, dir.y * s + point1_.y, dir.z * s + point1_.z);
		}
	}

	bool contain(const point_type &pt, const T &tol) const
	{
		if (!toPlane().contain(pt, tol)) return false;

		const vector_type &normal = getNormalVector();
		const T nor[3] = { normal.x >= 0 ? normal.x : -normal.x, normal.y >= 0 ? normal.y : -normal.y, normal.z >= 0 ? normal.z : -normal.z };

		const size_t idx = std::distance(nor, std::max_element(nor, nor + 3));
		switch (idx)
		{
		case 0:  // project onto yz-plane
			return Triangle2<T>(Triangle2<T>::point_type(point1_.y, point1_.z), Triangle2<T>::point_type(point2_.y, point2_.z), Triangle2<T>::point_type(point3_.y, point3_.z)).contain(Triangle2<T>::point_type(pt.y, pt.z), tol);
		case 1:  // project onto zx-plane
			return Triangle2<T>(Triangle2<T>::point_type(point1_.z, point1_.x), Triangle2<T>::point_type(point2_.z, point2_.x), Triangle2<T>::point_type(point3_.z, point3_.x)).contain(Triangle2<T>::point_type(pt.z, pt.x), tol);
		case 2:  // project onto xy-plane
			return Triangle2<T>(Triangle2<T>::point_type(point1_.x, point1_.y), Triangle2<T>::point_type(point2_.x, point2_.y), Triangle2<T>::point_type(point3_.x, point3_.y)).contain(Triangle2<T>::point_type(pt.x, pt.y), tol);
		default:
			throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);
		}
	}

	Plane3<T> toPlane() const  {  return Plane3<T>(point1_, point2_, point3_);  }

private:
	bool isEqual(const point_type &pt1, const point_type &pt2, const T &tol) const
	{  return (pt1.x-pt2.x >= -tol && pt1.x-pt2.x <= tol) && (pt1.y-pt2.y >= -tol && pt1.y-pt2.y <= tol) && (pt1.z-pt2.z >= -tol && pt1.z-pt2.z <= tol);  }

private:
	point_type point1_, point2_, point3_;
};

}  // namespace swl


#endif  //  __SWL_MATH__TRIANGLE__H_
