#if !defined(__SWL_MATH__QUATERNION_MATRIX__H_)
#define __SWL_MATH__QUATERNION_MATRIX__H_ 1


#include "swl/math/Quaternion.h"
#include "swl/math/RMatrix.h"
#include "swl/math/Vector.h"
#include "swl/math/MathConstant.h"
#include <iomanip>


namespace swl {

//-----------------------------------------------------------------------------------------
// class QMatrix: 3x3 quaternion matrix for rotation of 3D vector (3x1 vector)
// [  e0  e3  e6  ]
// [  e1  e4  e7  ]
// [  e2  e5  e8  ]

template<typename T>
class QMatrix
{
public:
    typedef T value_type;
    
public:
	QMatrix()
	: q_(T(1), T(0), T(0), T(0))
	{}
	explicit QMatrix(const Vector3<T> &rX, const Vector3<T> &rY, const Vector3<T> &rZ)
	: q_(Quaternion<T>::to_quaternion(RMatrix3<T>(rX, rY, rZ)))
	{}
	explicit QMatrix(const T rhs[9])
	: q_(Quaternion<T>::to_quaternion(RMatrix3<T>(rhs)))
	{}
	explicit QMatrix(const Quaternion<T> &rQuat)
	: q_(rQuat)
	{}
	explicit QMatrix(const RMatrix3<T> &rRMat)
	: q_(Quaternion<T>::to_quaternion(rRMat))
	{}
	QMatrix(const QMatrix &rhs)
	: q_(rhs.q_)
	{}
	~QMatrix()  {}

	QMatrix & operator=(const QMatrix &rhs)
	{
		if (this == &rhs) return *this;
		q_ = rhs.q_;
		return *this;
	}

public:
	///
	Quaternion<T> & q()  {  return q_;  }
	const Quaternion<T> & q() const  {  return q_;  }

	///
	bool get(T entry[9]) const 
	{  return RMatrix3<T>::toRotationMatrix(q_).get(entry);  }
	bool set(const T entry[9])
	{
		q_ = Quaternion<T>::to_quaternion(RMatrix3<T>(entry));
		return isValid();
	}

	///
	bool isValid(const T &tTol = (T)MathConstant::EPS) const
	{  return !q_.isZero(tTol) && q_.isUnit(tTol);  }
	bool isEqual(const QMatrix &rhs, const T &tTol = (T)MathConstant::EPS) const
	{  return q_.isEqual(rhs.q_, tTol);  }

	/// comparison operator
    bool operator==(const QMatrix &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const QMatrix &rhs) const  {  return !isEqual(rhs);  }

	///
	QMatrix operator*(const QMatrix &rhs) const
	{  return QMatrix(q_ * rhs.q_);  }
	QMatrix & operator*=(const QMatrix &rhs)
	{  q_ *= rhs.q_;  return *this;  }

	/// rotation
	Vector3<T> operator*(const Vector3<T> &rV) const
	{
		Quaternion<T> aQuat(T(0), rV.x(), rV.y(), rV.z()), aUnit(q_.unit());
		aQuat = aUnit * aQuat * aUnit.conjugate();
		return Vector3<T>(aQuat.q1(), aQuat.q2(), aQuat.q3());
	}

	///
	void identity()
	{  q_.q0() = T(1);  q_.q1() = T(0);  q_.q2() = T(0);  q_.q3() = T(0);  }
	QMatrix transpose() const
	{  return inverse();  }
	QMatrix inverse() const
	{  return q_.conjugate();  }

	///
	bool orthonormalize()
	{
		if (q_.isZero()) return false;
		q_ = q_.unit();
		return true;
	}

	///
	static QMatrix toRotationMatrix(const T &rad, const Vector3<T> &axis);

private:
	Quaternion<T> q_;
};


template<typename T>
/*static*/ QMatrix<T> QMatrix<T>::toRotationMatrix(const T &rad, const Vector3<T> &axis)
{
	// for making a unit quaternion
	const T tNorm = (T)sqrt(axis.x()*axis.x() + axis.y()*axis.y() + axis.z()*axis.z());
	if (MathUtil::isZero(tNorm))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return QMatrix<T>();
	}

	QMatrix<T> quatMat;
	const T s = (T)sin(rad / T(2));
	quatMat.q_.q0() = (T)cos(rad / T(2));
	quatMat.q_.q1() = s * axis.x() / tNorm;
	quatMat.q_.q2() = s * axis.y() / tNorm;
	quatMat.q_.q3() = s * axis.z() / tNorm;

	return quatMat;
}


//-----------------------------------------------------------------------------------------
// 3D Quaternion Matrix API

template<typename T>
std::istream & operator>>(std::istream &stream, QMatrix<T> &mat)
{
	// [> q0  q1  q2  q3 <] means a quaternion matrix
	char ch;
	stream >> ch >> ch >> mat.q().q0() >> mat.q().q1() >> mat.q().q2() >> mat.q().q3() >> ch >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const QMatrix<T> &mat)
{
	// [> q0  q1  q2  q3 <] means a quaternion matrix
	stream << "[> " << mat.q().q0() << "  " << mat.q().q1() << "  " << mat.q().q2() << "  " << mat.q().q3() << " <]";
	return stream;
}

}  // namespace swl


#endif  // __SWL_MATH__QUATERNION_MATRIX__H_
