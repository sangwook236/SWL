#if !defined(__SWL_MATH__VECTOR__H_)
#define __SWL_MATH__VECTOR__H_ 1


#include "swl/math/MathUtil.h"
#include "swl/base/LogException.h"
#include "swl/base/Point.h"
#include <iostream>
#include <cmath>


namespace swl {

//----------------------------------------------------------------------------------------------
// class Vector2: 2D vector (2x1 vector)

template<typename T>
class Vector2
{
public:
    typedef T value_type;

public:
	Vector2(const T &tX = T(0), const T &tY = T(0))
	: x_(tX), y_(tY)
	{}
	explicit Vector2(const T rhs[2])
	: x_(rhs[0]), y_(rhs[1])
	{}
	explicit Vector2(const Point2<T> &pt)
	: x_(pt.x), y_(pt.y)
	{}
	Vector2(const Vector2 &rhs)
	: x_(rhs.x_), y_(rhs.y_)
	{}
	~Vector2()  {}

	Vector2 & operator=(const Vector2 &rhs)
	{
	    if (this == &rhs) return *this;
	    x_ = rhs.x_;  y_ = rhs.y_;
	    return *this;
	}

public:
	/// accessor & mutator
	T & x()  {  return x_;  }
	const T & x() const  {  return x_;  }
	T & y()  {  return y_;  }
	const T & y() const  {  return y_;  }

	///
	T & operator[](const int iIndex)
	{
		switch (iIndex)
		{
		case 0: return x_;
		case 1: return y_;
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iIndex < 0 ? x_ : y_;
		}
	}
	const T & operator[](const int iIndex) const
	{
		switch (iIndex)
		{
		case 0: return x_;
		case 1: return y_;
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iIndex < 0 ? x_ : y_;
		}
	}

	///
	bool isZero(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(x_, tTol) && MathUtil::isZero(y_, tTol);  }
	bool isEqual(const Vector2 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{  return (*this - rhs).isZero(tTol);  }
	bool isUnit(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(norm() - T(1), tTol);  }
	bool isOrthogonal(const Vector2 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		if (MathUtil::isZero(norm(), tTol) || MathUtil::isZero(rhs.norm(), tTol)) return false;
		return MathUtil::isZero(dot(rhs));
	}
	bool isParallel(const Vector2 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		if (MathUtil::isZero(norm(), tTol) || MathUtil::isZero(rhs.norm(), tTol)) return false;
		Vector2 unit1 = unit(), unit2 = rhs.unit();
		return (unit1 - unit2).isZero(tTol) || (unit1 + unit2).isZero(tTol);
	}

	/// comparison operator
    bool operator==(const Vector2 &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const Vector2 &rhs) const  {  return !isEqual(rhs);  }

	///
	Vector2 & operator+()  {  return *this;  }
	const Vector2 & operator+() const  {  return *this;  }
	Vector2 operator+(const Vector2 &rhs) const
	{  return Vector2(x_+rhs.x_, y_+rhs.y_);  }
	Vector2 & operator+=(const Vector2 &rhs)
	{  x_ += rhs.x_;  y_ += rhs.y_;  return *this;  }
	Vector2 operator-() const  {  return Vector2(-x_, -y_);  }
	Vector2 operator-(const Vector2 &rhs) const
	{  return Vector2(x_-rhs.x_, y_-rhs.y_);  }
	Vector2 & operator-=(const Vector2 &rhs)
	{  x_ -= rhs.x_;  y_ -= rhs.y_;  return *this;  }

	/// dot product
	T operator*(const Vector2 &rhs) const  {  return dot(rhs);  }

	/// scalar operation
	Vector2 operator*(const T &S) const
	{  return Vector2(x_*S, y_*S);  }
	Vector2 & operator*=(const T &S)
	{  x_ *= S;  y_ *= S;  return *this;  }
	Vector2 operator/(const T &S) const
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    return Vector2(x_/S, y_/S);
	}
	Vector2 & operator/=(const T &S)
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    x_ /= S;  y_ /= S;
	    return *this;
	}

	/// Euclidean norm or L2-norm: блVбл = (в▓|V(i)|^2)^(1/2)
	T norm() const
	{  return (T)::sqrt(x_*x_ + y_*y_);  }
	/// dot product
	T dot(const Vector2 &rhs) const
	{  return x_*rhs.x_ + y_*rhs.y_;  }
	/// unit vector
	Vector2 unit(const T &tTol = (T)MathConstant::EPS) const
	{
		const T tNorm(norm());
		if (MathUtil::isZero(tNorm, tTol))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		return *this / tNorm;
	}

	///
	void normalize(const T &tTol = (T)MathConstant::EPS)
	{
		const T tNorm(norm());
		if (MathUtil::isZero(tNorm, tTol))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return false;
		}
		*this /= tNorm;
	}
/*
	///
	Vector2 & translate(const T &tX, const T &tY);
	Vector2 & rotate(const T &tAngle);
	Vector2 & scale(const T &tX, const T &tY);
	Vector2 & mirror(const T &tAngle);
*/
	///
	void toArray(T array[2]) const
	{  array[0] = x_;  array[1] = y_;  }

	Point2<T> toPoint() const
	{  return Point2<T>(x_, y_);  }

private:
	T x_, y_;
};


//-----------------------------------------------------------------------------------------
// 2D Transformation Vector API

template<typename T>
std::istream & operator>>(std::istream &stream, Vector2<T> &v)
{
	// < x, y > means a transformation vector
	char ch;
	stream >> ch >> v.x() >> ch >> v.y() >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const Vector2<T> &v)
{
	// < x, y > means a transformation vector
	stream << "< " << v.x() << " , " << v.y() << " >";
	return stream;
}


//----------------------------------------------------------------------------------------------
// class HVector2: 2D homogeneous vector (2x1 vector)

template<typename T>
class HVector2
{
public:
    typedef T value_type;

public:
	HVector2(const T &tX = T(0), const T &tY = T(0), const T &tW = T(1))
	: x_(tX), y_(tY), w_(tW)
	{}
	explicit HVector2(const T rhs[3])
	: x_(rhs[0]), y_(rhs[1]), w_(rhs[2])
	{}
	HVector2(const HVector2 &rhs)
	: x_(rhs.x_), y_(rhs.y_), w_(rhs.w_)
	{}
	~HVector2()  {}

	HVector2 & operator=(const HVector2 &rhs)
	{
	    if (this == &rhs) return *this;
	    x_ = rhs.x_;  y_ = rhs.y_;  w_ = rhs.w_;
	    return *this;
	}

public:
	/// accessor & mutator
	T & x()  {  return x_;  }
	const  T& x() const  {  return x_;  }
	T & y()  {  return y_;  }
	const T & y() const  {  return y_;  }
	T & w()  {  return w_;  }
	const T & w() const  {  return w_;  }

	///
	void toArray(T array[3]) const
	{  array[0] = x_;  array[1] = y_;  array[2] = w_;  }

private:
	T x_, y_, w_;
};


//-----------------------------------------------------------------------------------------
// 2D Homogeneous Vector API

template<typename T>
std::istream & operator>>(std::istream &stream, HVector2<T> &v)
{
	// < x, y, w > means a homogeneous vector
	char ch;
	stream >> ch >> v.x() >> ch >> v.y() >> ch >> v.w() >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const HVector2<T> &v)
{
	// < x, y, w > means a homogeneous vector
	stream << "< " << v.x() << " , " << v.y() << " , " << v.w() << " >";
	return stream;
}


//----------------------------------------------------------------------------------------------
// class Vector3: 3D vector (3x1 vector)

template<typename T>
class Vector3
{
public:
    typedef T value_type;

public:
	Vector3(const T &tX = T(0), const T &tY = T(0), const T &tZ = T(0))
	: x_(tX), y_(tY), z_(tZ)
	{}
	explicit Vector3(const T rhs[3])
	: x_(rhs[0]), y_(rhs[1]), z_(rhs[2])
	{}
	explicit Vector3(const Point3<T> &pt)
	: x_(pt.x), y_(pt.y), z_(pt.z)
	{}
	Vector3(const Vector3 &rhs)
	: x_(rhs.x_), y_(rhs.y_), z_(rhs.z_)
	{}
	~Vector3()  {}

	Vector3 & operator=(const Vector3 &rhs)
	{
	    if (this == &rhs) return *this;
	    x_ = rhs.x_;  y_ = rhs.y_;  z_ = rhs.z_;
	    return *this;
	}

public:
	/// accessor & mutator
	T & x()  {  return x_;  }
	const T & x() const  {  return x_;  }
	T & y()  {  return y_;  }
	const T & y() const  {  return y_;  }
	T & z()  {  return z_;  }
	const T & z() const  {  return z_;  }

	///
	T & operator[](const int iIndex)
	{
		switch (iIndex)
		{
		case 0: return x_;
		case 1: return y_;
		case 2: return z_;
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iIndex < 0 ? x_ : z_;
		}
	}
	const T & operator[](const int iIndex) const
	{
		switch (iIndex)
		{
		case 0: return x_;
		case 1: return y_;
		case 2: return z_;
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iIndex < 0 ? x_ : z_;
		}
	}

	///
	bool isZero(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(x_, tTol) && MathUtil::isZero(y_, tTol) && MathUtil::isZero(z_, tTol);  }
	bool isEqual(const Vector3 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{  return (*this - rhs).isZero(tTol);  }
	bool isUnit(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(norm() - T(1), tTol);  }
	bool isOrthogonal(const Vector3 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		if (MathUtil::isZero(norm(), tTol) || MathUtil::isZero(rhs.norm(), tTol)) return false;
		return MathUtil::isZero(dot(rhs), tTol);
	}
	bool isParallel(const Vector3 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		if (MathUtil::isZero(norm(), tTol) || MathUtil::isZero(rhs.norm(), tTol)) return false;
#if 0
		const Vector3 unit1 = unit(), unit2 = rhs.unit();
		return (unit1 - unit2).isZero(tTol) || (unit1 + unit2).isZero(tTol);
#else
		return cross(rhs).isZero(tTol);
#endif
	}

	/// comparison operator
    bool operator==(const Vector3 &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const Vector3 &rhs) const  {  return !isEqual(rhs);  }

	///
	Vector3 & operator+()  {  return *this;  }
	const Vector3 & operator+() const  {  return *this;  }
	Vector3 operator+(const Vector3 &rhs) const
	{  return Vector3(x_+rhs.x_, y_+rhs.y_, z_+rhs.z_);  }
	Vector3 & operator+=(const Vector3 &rhs)
	{  x_ += rhs.x_;  y_ += rhs.y_;  z_ += rhs.z_;  return *this;  }
	Vector3 operator-() const  {  return Vector3(-x_, -y_, -z_);  }
	Vector3 operator-(const Vector3 &rhs) const
	{  return Vector3(x_-rhs.x_, y_-rhs.y_, z_-rhs.z_);  }
	Vector3 & operator-=(const Vector3 &rhs)
	{  x_ -= rhs.x_;  y_ -= rhs.y_;  z_ -= rhs.z_;  return *this;  }

	/// dot product
	T operator*(const Vector3 &rhs) const  {  return dot(rhs);  }

	/// scalar operation
	Vector3 operator*(const T &S) const
	{  return Vector3(x_*S, y_*S, z_*S);  }
	Vector3 & operator*=(const T &S)
	{  x_ *= S;  y_ *= S;  z_ *= S;  return *this;  }
	Vector3 operator/(const T &S) const
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    return Vector3(x_/S, y_/S, z_/S);
	}
	Vector3 & operator/=(const T &S)
	{
		if (MathUtil::isZero(S))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
	    x_ /= S;  y_ /= S;  z_ /= S;
	    return *this;
	}

	/// Euclidean norm or L2-norm: блVбл = (в▓|V(i)|^2)^(1/2)
	T norm() const
	{  return (T)::sqrt(x_*x_ + y_*y_ + z_*z_);  }
	/// dot product
	T dot(const Vector3 &rhs) const
	{  return x_*rhs.x_ + y_*rhs.y_ + z_*rhs.z_;  }
	/// cross product
	Vector3 cross(const Vector3 &rhs) const
	{  return Vector3(y_*rhs.z_ - z_*rhs.y_, z_*rhs.x_ - x_*rhs.z_, x_*rhs.y_ - y_*rhs.x_);  }
	/// unit vector
	Vector3 unit(const T &tTol = (T)MathConstant::EPS) const
	{
		const T tNorm(norm());
		if (MathUtil::isZero(tNorm, tTol))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return *this;
		}
		return *this / tNorm;
	}

	///
	void normalize(const T &tTol = (T)MathConstant::EPS)
	{
		const T tNorm(norm());
		if (MathUtil::isZero(tNorm, tTol))
		{
			throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
			//return false;
		}
		*this /= tNorm;
	}
/*
	///
	Vector2<T> & translate(const T &tX, const T &tY);
	Vector2<T> & rotate(const T &tAngle);
	Vector2<T> & scale(const T &tX, const T &tY);
	Vector2<T> & mirror(const T &tAngle);
*/
	///
	void toArray(T array[3]) const
	{  array[0] = x_;  array[1] = y_;  array[2] = z_;  }

	Point3<T> toPoint() const
	{  return Point3<T>(x_, y_, z_);  }

private:
	T x_, y_, z_;
};


//-----------------------------------------------------------------------------------------
// 3D Transformation Vector API

template<typename T>
std::istream & operator>>(std::istream &stream, Vector3<T> &v)
{
	// < x, y, z > means a transformation vector
	char ch;
	stream >> ch >> v.x() >> ch >> v.y() >> ch >> v.z() >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const Vector3<T> &v)
{
	// < x, y, z > means a transformation vector
	stream << "< " << v.x() << " , " << v.y() << " , " << v.z() << " >";
	return stream;
}


//----------------------------------------------------------------------------------------------
// class HVector3: 3D homogeneous vector (4x1 vector)

template<typename T>
class HVector3
{
public:
    typedef T value_type;

public:
	HVector3(const T &tX = T(0), const T &tY = T(0), const T &tZ = T(0), const T &tW = T(1))
	: x_(tX), y_(tY), z_(tZ), w_(tW)
	{}
	explicit HVector3(const T rhs[4])
	: x_(rhs[0]), y_(rhs[1]), z_(rhs[2]), w_(rhs[3])
	{}
	HVector3(const HVector3 &rhs)
	: x_(rhs.x_), y_(rhs.y_), z_(rhs.z_), w_(rhs.w_)
	{}
	~HVector3()  {}

	HVector3 & operator=(const HVector3 &rhs)
	{
	    if (this == &rhs) return *this;
	    x_ = rhs.x_;  y_ = rhs.y_;  z_ = rhs.z_;  w_ = rhs.w_;
	    return *this;
	}

public:
	/// accessor & mutator
	T & x()  {  return x_;  }
	const T & x() const  {  return x_;  }
	T & y()  {  return y_;  }
	const T & y() const  {  return y_;  }
	T & z()  {  return z_;  }
	const T & z() const  {  return z_;  }
	T & w()  {  return w_;  }
	const T & w() const  {  return w_;  }

	///
	void toArray(T array[4]) const
	{  array[0] = x_;  array[1] = y_;  array[2] = z_;  array[3] = w_;  }

private:
	T x_, y_, z_, w_;
};


//-----------------------------------------------------------------------------------------
// 3D Homogeneous Vector API

template<typename T>
std::istream& operator>>(std::istream& stream, HVector3<T>& v)
{
	// < x, y, z, w > means a homogeneous vector
	char ch;
	stream >> ch >> v.x() >> ch >> v.y() >> ch >> v.z() >> ch >> v.w() >> ch;
	return stream;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const HVector3<T>& v)
{
	// < x, y, z, w > means a homogeneous vector
	stream << "< " << v.x() << " , " << v.y() << " , " << v.z() << " , " << v.w() << " >";
	return stream;
}

}  // namespace swl


#endif  // __SWL_MATH__VECTOR__H_
