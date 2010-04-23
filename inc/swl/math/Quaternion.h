#if !defined(__SWL_MATH__QUATERNION__H_)
#define __SWL_MATH__QUATERNION__H_ 1


#include "swl/math/RMatrix.h"
#include "swl/base/LogException.h"
#include <cmath>


namespace swl {

template<typename T> class RMatrix3;


//-----------------------------------------------------------------------------------------
// class Quaternion: quaternion, q0 + q1*{i} + q2*{j} + q3*{k}

template<typename T>
class Quaternion
{
public:
    typedef T value_type;
    
public:
	Quaternion(const T &tQ0 = T(0), const T &tQ1 = T(0), const T &tQ2 = T(0), const T &tQ3 = T(0))
	: q0_(tQ0), q1_(tQ1), q2_(tQ2), q3_(tQ3)
	{}
	explicit Quaternion(const T rhs[4])
	: q0_(rhs[0]), q1_(rhs[1]), q2_(rhs[2]), q3_(rhs[3])
    {}
	Quaternion(const Quaternion &rhs)
	: q0_(rhs.q0_), q1_(rhs.q1_), q2_(rhs.q2_), q3_(rhs.q3_)
	{}
	~Quaternion()  {}

	Quaternion & operator=(const Quaternion &rhs)
    {
    	if (this == &rhs) return *this;
    	q0_ = rhs.q0_;  q1_ = rhs.q1_;  q2_ = rhs.q2_;  q3_ = rhs.q3_;
    	return *this;
    }

public:
	/// accessor & mutator
	T & q0()  {  return q0_;  }
	const T & q0() const  {  return q0_;  }
	T & q1()  {  return q1_;  }
	const T & q1() const  {  return q1_;  }
	T & q2()  {  return q2_;  }
	const T & q2() const  {  return q2_;  }
	T & q3()  {  return q3_;  }
	const T & q3() const  {  return q3_;  }

	///
    T & operator[](const unsigned int iIndex);
    const T & operator[](const unsigned int iIndex) const;

	///
	bool isZero(const T &tTol = (T)MathConstant::EPS) const
	{
		return MathUtil::isZero(q0_, tTol) && MathUtil::isZero(q1_, tTol) &&
			   MathUtil::isZero(q2_, tTol) && MathUtil::isZero(q3_, tTol);
	}
	bool isEqual(const Quaternion &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		return MathUtil::isZero(q0_ - rhs.q0_, tTol) && MathUtil::isZero(q1_ - rhs.q1_, tTol) &&
			   MathUtil::isZero(q2_ - rhs.q2_, tTol) && MathUtil::isZero(q3_ - rhs.q3_, tTol);
	}
	bool isUnit(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(norm() - T(1), tTol);  }
	bool isScalar(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(q1_, tTol) && MathUtil::isZero(q2_, tTol) && MathUtil::isZero(q3_, tTol);  }
	bool isVector(const T &tTol = (T)MathConstant::EPS) const
	{  return MathUtil::isZero(q0_, tTol);  }

	/// comparison operator
	bool operator==(const Quaternion &rhs) const
	{  return isEqual(rhs);  }
	bool operator!=(const Quaternion &rhs) const
	{  return !isEqual(rhs);  }

	/// arithmetic functions
	Quaternion & operator+()  {  return *this;  }
	const Quaternion & operator+() const  {  return *this;  }
	Quaternion operator+(const Quaternion &rhs) const
	{  return Quaternion(q0_+rhs.q0_, q1_+rhs.q1_, q2_+rhs.q2_, q3_+rhs.q3_);  }
	Quaternion & operator+=(const Quaternion &rhs)
	{  q0_ += rhs.q0_;  q1_ += rhs.q1_;  q2_ += rhs.q2_;  q3_ += rhs.q3_;  return *this;  }
	Quaternion operator-() const
	{  return Quaternion(-q0_, -q1_, -q2_, -q3_);  }
	Quaternion operator-(const Quaternion &rhs) const
	{  return Quaternion(q0_-rhs.q0_, q1_-rhs.q1_, q2_-rhs.q2_, q3_-rhs.q3_);  }
	Quaternion & operator-=(const Quaternion &rhs)
	{  q0_ -= rhs.q0_;  q1_ -= rhs.q1_;  q2_ -= rhs.q2_;  q3_ -= rhs.q3_;  return *this;  }
	Quaternion operator*(const Quaternion &rhs) const;
	Quaternion & operator*=(const Quaternion &rhs)
	{  return *this = *this * rhs;  }
	Quaternion operator/(const Quaternion &rhs) const
	{  return *this * rhs.inverse();  }
	Quaternion & operator/=(const Quaternion &rhs)
	{  return *this *= rhs.inverse();  }

	/// scalar operation
	Quaternion operator*(const T &S) const
    {  return Quaternion(q0_*S, q1_*S, q2_*S, q3_*S);  }
	Quaternion & operator*=(const T &S)
	{  q0_ *= S;  q1_ *= S;  q2_ *= S;  q3_ *= S;  return *this;  }
	Quaternion operator/(const T &S) const;
	Quaternion & operator/=(const T& S);
/*
	friend Quaternion operator*(const T &S, const Quaternion &rhs)
    {  return rhs * S;  }
	friend Quaternion operator/(const T &S, const Quaternion &rhs)
    {  return rhs / S;  }
*/
	/// norm of a quaternion
	T norm() const
	{  return q0_*q0_ + q1_*q1_ + q2_*q2_ + q3_*q3_;  }
	/// conjugate of a quaternion
	Quaternion conjugate() const
    {  return Quaternion(q0_, -q1_, -q2_, -q3_);  }
	/// inverse(reciprocal) of a quaternion
	Quaternion inverse() const;

	/// unit quaternion
	Quaternion unit() const;
	/// scalar part of a quaternion
	Quaternion scalar() const
	{  return Quaternion(q0_, T(0), T(0), T(0));  }
	void scalar(T& S) const  {  S = q0_;  }
	/// vector part of a quaternion
	Quaternion vector() const
	{  return Quaternion(T(0), q1_, q2_, q3_);  }
	void vector(T aVector[3]) const
	{  aVector[0] = q1_;  aVector[1] = q2_;  aVector[2] = q3_;  }

	///
	Quaternion rotate(const Quaternion &quat);
	Quaternion rotate(const T &rad, const Vector3<T> &axis);

	/// angle-axis representation
	T angle();
	Vector3<T> axis();

	/// angle-axis representation ==> quaternion
	static Quaternion toQuaternion(const T &rad, const Vector3<T> &axis);

	/// 3x3 rotation matirx ==> quaternion
	static Quaternion toQuaternion(const RMatrix3<T> &rotMat);

	/// slerp: spherical linear interpolation
	/// 0 <= t <= 1 && uq0, uq1: unit quaternions
	static Quaternion slerp(const T &t, const Quaternion &uq0, const Quaternion &uq1);

	/// squad: spherical cubic interpolation
	/// 0 <= t <= 1 && uq0, uq1, uq2, uq3: unit quaternions
	static Quaternion squad(const T &t, const Quaternion &uq0, const Quaternion &uq1, const Quaternion &uq2, const Quaternion &uq3);

private:
	/// for unit quaternion
	Quaternion pow(const T &t);
	Quaternion log();

private:
	/// the real and 3-vector parts of a quaternion: q0 + q1*{i} + q2*{j} + q3*{k}
	T q0_, q1_, q2_, q3_;
};


template<typename T>
T & Quaternion<T>::operator[](const unsigned int iIndex)
{
	switch (iIndex)
	{
	case 0: return q0_;
	case 1: return q1_;
	case 2: return q2_;
	case 3: return q3_;
	default:
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
		//return iIndex < 0 ? q0_ : q3_;
	}
}

template<typename T>
const T & Quaternion<T>::operator[](const unsigned int iIndex) const
{
	switch (iIndex)
	{
	case 0: return q0_;
	case 1: return q1_;
	case 2: return q2_;
	case 3: return q3_;
	default:
		throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
		//return iIndex < 0 ? q0_ : q3_;
	}
}

template<typename T>
Quaternion<T> Quaternion<T>::operator*(const Quaternion<T> &rhs) const
{
	return Quaternion<T>(
		q0_*rhs.q0_ - q1_*rhs.q1_ - q2_*rhs.q2_ - q3_*rhs.q3_,
    	q0_*rhs.q1_ + q1_*rhs.q0_ + q2_*rhs.q3_ - q3_*rhs.q2_,
    	q0_*rhs.q2_ + q2_*rhs.q0_ + q3_*rhs.q1_ - q1_*rhs.q3_,
    	q0_*rhs.q3_ + q3_*rhs.q0_ + q1_*rhs.q2_ - q2_*rhs.q1_
    );
}

template<typename T>
Quaternion<T> Quaternion<T>::operator/(const T &S) const
{
	if (MathUtil::isZero(S))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	return Quaternion<T>(q0_/S, q1_/S, q2_/S, q3_/S);
}

template<typename T>
Quaternion<T> & Quaternion<T>::operator/=(const T &S)
{
	if (MathUtil::isZero(S))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	q0_ /= S;  q1_ /= S;  q2_ /= S;  q3_ /= S;
	return *this;
}

template<typename T>
Quaternion<T> Quaternion<T>::inverse() const
{
	const T tNorm = norm();
	// nearly zero quaternion
	if (MathUtil::isZero(tNorm))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	return Quaternion<T>(q0_/tNorm, -q1_/tNorm, -q2_/tNorm, -q3_/tNorm);
}

template<typename T>
Quaternion<T> Quaternion<T>::unit() const
{
	const T h = (T)sqrt(norm());
	// nearly zero quaternion
	if (MathUtil::isZero(h))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return *this;
	}
	return Quaternion<T>(q0_/h, q1_/h, q2_/h, q3_/h);
}

template<typename T>
Quaternion<T> Quaternion<T>::rotate(const Quaternion<T> &quat)
{  return quat * *this * quat.inverse();  }

template<typename T>
Quaternion<T> Quaternion<T>::rotate(const T &rad, const Vector3<T> &axis)
{  return rotate(Quaternion<T>::toQuaternion(rad, axis));  }

template<typename T>
T Quaternion<T>::angle()
{
	const T h = (T)sqrt(norm());
	if (MathUtil::isZero(h))  // nearly zero quaternion
	{
		//throw LogException(LogException::L_ERROR, "unknown", __FILE__, __LINE__, __FUNCTION__);
		return T(0);
	}

	const T h2 = (T)sqrt(q1_*q1_ + q2_*q2_ + q3_*q3_);
	if (MathUtil::isZero(q0_))  // nearly pure vector
		return T(2) * (T)asin(h2 / h);
	else
		return T(2) * (T)atan2(h2, q0_);
}

template<typename T>
Vector3<T> Quaternion<T>::axis()
{
	const T h2 = (T)sqrt(q1_*q1_ + q2_*q2_ + q3_*q3_);
	if (MathUtil::isZero(h2))
	{
		//throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		return Vector3<T>();
	}

	return Vector3<T>(q1_ / h2, q2_ / h2, q3_ / h2);
}

template<typename T>
/*static*/ Quaternion<T> Quaternion<T>::toQuaternion(const T &rad, const Vector3<T> &axis)
{
	// for making a unit quaternion
	const T tNorm = (T)sqrt(axis.x()*axis.x() + axis.y()*axis.y() + axis.z()*axis.z());
	if (MathUtil::isZero(tNorm))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return Quaternion<T>();
	}

	const T s = (T)sin(rad / T(2));
	const T tmp = s / tNorm;
	return Quaternion<T>((T)cos(rad / T(2)), axis.x() * tmp, axis.y() * tmp, axis.z() * tmp);
}

// 3x3 rotation matirx ==> quaternion
template<typename T>
/*static*/ Quaternion<T> Quaternion<T>::toQuaternion(const RMatrix3<T> &rotMat)
//  R = [  e0  e3  e6  ]
//      [  e1  e4  e7  ]
//      [  e2  e5  e8  ]
{
	const T tTrace = rotMat[0] + rotMat[4] + rotMat[8];
	if (tTrace < T(-1))
	{
		throw LogException(LogException::L_ERROR, "domain error", __FILE__, __LINE__, __FUNCTION__);
		//return Quaternion<T>();
	}

	Quaternion<T> quat;
	// case 1
	quat.q0_ = (T)sqrt((tTrace + T(1)) / T(4));
	// case 2
	//quat.q0_ = (T)-sqrt((tTrace + T(1)) / T(4));

	if (MathUtil::isZero(quat.q0_))
	{
		// case 1
		quat.q1_ = (T)sqrt((T(1) + T(2)*rotMat[0] - tTrace) / T(4));
		// case 2
		//quat.q1_ = -(T)sqrt((T(1) + T(2)*rotMat[0] - tTrace) / T(4));

		if ((rotMat[1]+rotMat[3])*quat.q1_ >= T(0))
			quat.q2_ = (T)sqrt((T(1) + T(2)*rotMat[4] - tTrace) / T(4));
		else
			quat.q2_ = -(T)sqrt((T(1) + T(2)*rotMat[4] - tTrace) / T(4));

		if ((rotMat[2]+rotMat[6])*quat.q1_ >= T(0))
			quat.q3_ = (T)sqrt((T(1) + T(2)*rotMat[8] - tTrace) / T(4));
		else
			quat.q3_ = -(T)sqrt((T(1) + T(2)*rotMat[8] - tTrace) / T(4));
	}
	else
	{
		const T tVal(T(4) * quat.q0_);
		// case 1
		quat.q1_ = (rotMat[5] - rotMat[7]) / tVal;
		quat.q2_ = (rotMat[6] - rotMat[2]) / tVal;
		quat.q3_ = (rotMat[1] - rotMat[3]) / tVal;
/*
		// case 2
		quat.q1_ = (rotMat[7] - rotMat[5]) / tVal;
		quat.q2_ = (rotMat[2] - rotMat[6]) / tVal;
		quat.q3_ = (rotMat[3] - rotMat[1]) / tVal;
*/
	}

	return quat;
}

// slerp: spherical linear interpolation
// 0 <= t <= 1 && q0, q1: unit quaternions
template<typename T>
/*static*/ Quaternion<T> Quaternion<T>::slerp(const T &t, const Quaternion<T> &uq0, const Quaternion<T> &uq1)
{
/*
	// for non-unit quaternions
	const T theta(???);  // an angle between uq0 & uq1
	const T s(sin(theta));
	if (MathUtil::isZero(s))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return Quaternion<T>();
	}

	return uq0 * (T)sin((T(1) - t) * theta) / s + uq1 * (T)sin(t * theta) / s;
*/
	// for unit quaternions
	return uq0 * (uq0.inverse() * uq1).pow(t);
}

// squad: spherical cubic interpolation
// 0 <= t <= 1 && uq0, uq1, uq2, uq3: unit quaternions
template<typename T>
/*static*/ Quaternion<T> Quaternion<T>::squad(const T &t, const Quaternion<T> &uq0, const Quaternion<T> &uq1, const Quaternion<T> &uq2, const Quaternion<T> &uq3)
{
/*
	// for non-unit quaternions
	return Quaternion<T>::slerp(T(2) * t * (T(1) - t), Quaternion<T>::slerp(t, uq0, uq3), Quaternion<T>::slerp(t, uq1, uq2));
*/
	// for unit quaternions
	return Quaternion<T>::slerp(t, uq0, uq3) * (Quaternion<T>::slerp(t, uq0, uq3).inverse() * Quaternion<T>::slerp(t, uq1, uq2)).pow(T(2) * t * (T(1) - t));
}

// for unit quaternion
template<typename T>
Quaternion<T> Quaternion<T>::pow(const T &t)
{
	const T theta(angle());
	const Vector3<T> U(axis());

	const T s(sin(t * theta)), c(cos(t * theta));
	return Quaternion<T>(c, s * U.x(), s * U.y(), s * U.z());
}

// for unit quaternion
template<typename T>
Quaternion<T> Quaternion<T>::log()
{
	const T theta(angle());
	const Vector3<T> U(axis());

	return Quaternion<T>(T(0), theta * U.x(), theta * U.y(), theta * U.z());
}


//-----------------------------------------------------------------------------------------
// Quaternion API

template<typename T>
std::istream & operator>>(std::istream &stream, Quaternion<T> &q)
{
	// < q0 + q1 {i} + q2 {j} + q3 {k} > means a quaternion
/*
	char ch, buf[16];
	stream >> ch >> q.q0() >> ch >> q.q1() >> buf >> ch >> q.q2() >> buf >> ch >> q.q3() >> buf >> ch;
*/
	char ch;
	stream >> ch
		   >> q.q0()
		   >> ch >> q.q1() >> ch >> ch >> ch
		   >> ch >> q.q2() >> ch >> ch >> ch
		   >> ch >> q.q3() >> ch >> ch >> ch
		   >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const Quaternion<T> &q)
{
	// < q0 + q1 {i} + q2 {j} + q3 {k} > means a quaternion
	stream << "< " << q.q0() << " + " << q.q1() << " {i} + " << q.q2() << " {j} + " << q.q3() << " {k} >";
	return stream;
}

}  // namespace swl


#endif  // __SWL_MATH__QUATERNION__H_
