#if !defined(__SWL_MATH__ROTATION_MATRIX__H_)
#define __SWL_MATH__ROTATION_MATRIX__H_ 1


#include "swl/math/Vector.h"
#include "swl/math/Quaternion.h"
#include "swl/math/MathConstant.h"
#include <iomanip>


namespace swl {

template<typename T> class Vector2;
template<typename T> class Vector3;
template<typename T> class Quaternion;


//-----------------------------------------------------------------------------------------
// class RMatrix2: 2x2 matrix for rotation of 2D vector ( 2x1 vector )
// [  Xx  Yx  ]  =  [  X  Y  ]  =  [  e0  e2  ]
// [  Xy  Yy  ]                    [  e1  e3  ]

template<typename T>
class RMatrix2
{
public:
    typedef T				value_type;
	typedef Vector2<T>		column_type;

public:
	RMatrix2()
	: X_(T(1), T(0)), Y_(T(0), T(1))
	{}
	explicit RMatrix2(const column_type &rX, const column_type &rY)
	: X_(rX), Y_(rY)
	{}
	explicit RMatrix2(const T rhs[4])
	: X_(rhs[0], rhs[1]), Y_(rhs[2], rhs[3])
	{}
	RMatrix2(const RMatrix2 &rhs)
	: X_(rhs.X_), Y_(rhs.Y_)
	{}
	~RMatrix2()  {}

	RMatrix2 & operator=(const RMatrix2 &rhs)
	{
		if (this == &rhs) return *this;
		X_ = rhs.X_;  Y_ = rhs.Y_;
		return *this;
	}

public:
	///
	column_type & X()  {  return X_;  }
	const column_type & X() const  {  return X_;  }
	column_type & Y()  {  return Y_;  }
	const column_type & Y() const  {  return Y_;  }

	///
	T operator[](const int iIndex) const  {  return getEntry(iIndex%2, iIndex/2);  }
	T operator()(const int iRow, const int iCol) const  {  return getEntry(iRow, iCol);  }
	
	///
	bool set(const T entry[4])
	{
		X_.x() = entry[0];  X_.y() = entry[1];
		Y_.x() = entry[2];  Y_.y() = entry[3];
		return isValid();
	}
	bool get(T entry[4]) const 
	{
		entry[0] = X_.x();  entry[1] = X_.y();
		entry[2] = Y_.x();  entry[3] = Y_.y();
		return true;
	}

	///
	bool isValid(const T &tTol = (T)MathConstant::EPS) const
	{  return X_.isUnit(tTol) && Y_.isUnit(tTol) && X_.isOrthogonal(Y_, tTol);  }
	bool isEqual(const RMatrix2 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{  return X_.isEqual(rhs.X_, tTol) && Y_.isEqual(rhs.Y_, tTol);  }

	/// comparison operator
    bool operator==(const RMatrix2 &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const RMatrix2 &rhs) const  {  return !isEqual(rhs);  }

	///
	RMatrix2 operator*(const RMatrix2 &rhs) const
	{
		return RMatrix2(
			X_*rhs.X_.x() + Y_*rhs.X_.y(),
			X_*rhs.Y_.x() + Y_*rhs.Y_.y()
		);
	}
	RMatrix2 & operator*=(const RMatrix2 &rhs)
	{  return *this = *this * rhs;  }
	column_type operator*(const column_type& rV) const
	{  return column_type(X_*rV.x() + Y_*rV.y());  }

	///
	void identity()
	{
		X_.x() = T(1);  X_.y() = T(0);
		Y_.x() = T(0);  Y_.y() = T(1);
	}
	RMatrix2 transpose() const
	{
		return RMatrix2(
			column_type(X_.x(), Y_.x()), 
			column_type(X_.y(), Y_.y())
		);
	}
	RMatrix2 inverse() const
	{  return transpose();  }

	///
	bool orthonormalize()
	{
		if (X_.isZero() || Y_.isZero()) return false;
		Y_ = column_type(-X_.y(), X_.x());
		X_ = X_.unit();
		Y_ = Y_.unit();
		return true;
	}

	/// angle-axis representation
	T angle() const
	{
		if (MathUtil::isZero(X_.x()))
			return (T)std::asin(X_.y()) >= T(0) ? (T)MathConstant::PI_2 : (T)-MathConstant::PI_2;
		else return (T)std::atan2(X_.y(), X_.x());
	}
	Vector3<T> axis() const
	{  return Vector3<T>(T(0), T(0), T(1));  }

protected:
	T getEntry(const int iRow, const int iCol) const
	{
		switch (iCol)
		{
		case 0:
			switch (iRow)
			{
			case 0:  return X_.x();
			case 1:  return X_.y();
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return iRow < 0 ? X_.x() : X_.y();
			}
		case 1:
			switch (iRow)
			{
			case 0:  return Y_.x();
			case 1:  return Y_.y();
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return iRow < 0 ? Y_.x() : Y_.y();
			}
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iCol < 0 ? X_.x() : Y_.y();
		}
	}

private:
	column_type X_;
	column_type Y_;
};


//-----------------------------------------------------------------------------------------
// 2D Rotation Matrix API

template<typename T>
std::istream & operator>>(std::istream &stream, RMatrix2<T> &mat)
{
	// [> Xx  Yx  ] means a rotation matrix
	// [  Xy  Yy <]
	char ch;
	stream >> ch >> ch >> mat.X().x() >> mat.Y().x() >> ch
		   >> ch >> mat.X().y() >> mat.Y().y() >> ch >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const RMatrix2<T> &mat)
{
	// [> Xx  Yx  ] means a rotation matrix
	// [  Xy  Yy <]
	bool bIsNewLineInserted = false;
	if (bIsNewLineInserted)
	{
		const int nWidth = 8;
		stream << "[> " << std::setw(nWidth) << mat.X().x() << std::setw(nWidth) << mat.Y().x() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().y() << std::setw(nWidth) << mat.Y().y() << " <]";
	}
	else
	{
		stream << "[> " << mat.X().x() << "  " << mat.Y().x() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().y() << "  " << mat.Y().y() << " <]";
	}
	return stream;
}


//-----------------------------------------------------------------------------------------
// class RMatrix3: 3x3 matrix for rotation of 3D vector ( 3x1 vector )
// [  Xx  Yx  Zx  ]  =  [  X  Y  Z  ]  =  [  e0  e3  e6  ]
// [  Xy  Yy  Zy  ]                       [  e1  e4  e7  ]
// [  Xz  Yz  Zz  ]					   [  e2  e5  e8  ]

template<typename T>
class RMatrix3
{
public:
    typedef T				value_type;
	typedef Vector3<T>		column_type;

public:
	RMatrix3()
	: X_(T(1), T(0), T(0)), Y_(T(0), T(1), T(0)), Z_(T(0), T(0), T(1))
	{}
	explicit RMatrix3(const column_type &rX, const column_type &rY, const column_type &rZ)
	: X_(rX), Y_(rY), Z_(rZ)
	{}
	explicit RMatrix3(const T rhs[9])
	: X_(rhs[0], rhs[1], rhs[2]), Y_(rhs[3], rhs[4], rhs[5]), Z_(rhs[6], rhs[7], rhs[8])
	{}
	RMatrix3(const RMatrix3 &rhs)
	: X_(rhs.X_), Y_(rhs.Y_), Z_(rhs.Z_)
	{}
	~RMatrix3()  {}

	RMatrix3 & operator=(const RMatrix3 &rhs)
	{
		if (this == &rhs) return *this;
		X_ = rhs.X_;  Y_ = rhs.Y_;  Z_ = rhs.Z_;
		return *this;
	}

public:
	///
	column_type & X()  {  return X_;  }
	const column_type & X() const  {  return X_;  }
	column_type & Y()  {  return Y_;  }
	const column_type & Y() const  {  return Y_;  }
	column_type & Z()  {  return Z_;  }
	const column_type & Z() const  {  return Z_;  }

	///
	T operator[](const int iIndex) const  {  return getEntry(iIndex%3, iIndex/3);  }
	T operator()(const int iRow, const int iCol) const  {  return getEntry(iRow, iCol);  }
	
	///
	bool set(const T entry[9])
	{
		X_.x() = entry[0];  X_.y() = entry[1];  X_.z() = entry[2];
		Y_.x() = entry[3];  Y_.y() = entry[4];  Y_.z() = entry[5];
		Z_.x() = entry[6];  Z_.y() = entry[7];  Z_.z() = entry[8];
		return isValid();
	}
	bool get(T entry[9]) const 
	{
		entry[0] = X_.x();  entry[1] = X_.y();  entry[2] = X_.z();
		entry[3] = Y_.x();  entry[4] = Y_.y();  entry[5] = Y_.z();
		entry[6] = Z_.x();  entry[7] = Z_.y();  entry[8] = Z_.z();
		return true;
	}

	///
	bool isValid(const T &tTol = (T)MathConstant::EPS) const
	{
		return X_.isUnit(tTol) && Y_.isUnit(tTol) && Z_.isUnit(tTol) &&
			   X_.isOrthogonal(Y_, tTol) && X_.isOrthogonal(Z_, tTol) && Y_.isOrthogonal(Z_, tTol);
	}
	bool isEqual(const RMatrix3 &rhs, const T &tTol = (T)MathConstant::EPS) const
	{
		return X_.isEqual(rhs.X_, tTol) && Y_.isEqual(rhs.Y_, tTol) && Z_.isEqual(rhs.Z_, tTol);
	}

	/// comparison operator
    bool operator==(const RMatrix3 &rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const RMatrix3 &rhs) const  {  return !isEqual(rhs);  }

	///
	RMatrix3 operator*(const RMatrix3 &rhs) const
	{
		return RMatrix3(
			X_*rhs.X_.x() + Y_*rhs.X_.y() + Z_*rhs.X_.z(),
			X_*rhs.Y_.x() + Y_*rhs.Y_.y() + Z_*rhs.Y_.z(),
			X_*rhs.Z_.x() + Y_*rhs.Z_.y() + Z_*rhs.Z_.z()
		);
	}
	RMatrix3& operator*=(const RMatrix3 &rhs)
	{  return *this = *this * rhs;  }

	/// rotation
	column_type operator*(const column_type &rV) const
	{  return column_type(X_*rV.x() + Y_*rV.y() + Z_*rV.z());  }

	///
	void identity()
	{
		X_.x() = T(1);  X_.y() = T(0);  X_.z() = T(0);
		Y_.x() = T(0);  Y_.y() = T(1);  Y_.z() = T(0);
		Z_.x() = T(0);  Z_.y() = T(0);  Z_.z() = T(1);
	}
	RMatrix3 transpose() const
	{
		return RMatrix3(
			column_type(X_.x(), Y_.x(), Z_.x()), 
			column_type(X_.y(), Y_.y(), Z_.y()), 
			column_type(X_.z(), Y_.z(), Z_.z())
		);
	}
	RMatrix3 inverse() const
	{  return transpose();  }

	///
	bool orthonormalize()
	{
		if (X_.isZero() || Y_.isZero() || Z_.isZero()) return false;
		Z_ = X_.cross(Y_);
		Y_ = Z_.cross(X_);
		X_ = X_.unit();
		Y_ = Y_.unit();
		Z_ = Z_.unit();
		return true;
	}

	/// angle-axis representation
	T angle() const;
	Vector3<T> axis() const;

	/// angle-axis representation ==> 3x3 rotation matrix
	static RMatrix3 toRotationMatrix(const T &rad, const Vector3<T> &axis);

	/// quaternion ==> 3x3 rotation matrix
	static RMatrix3 toRotationMatrix(const Quaternion<T> &rQuat);

protected:
	T getEntry(const int iRow, const int iCol) const
	{
		switch (iCol)
		{
		case 0:
			switch (iRow)
			{
			case 0:  return X_.x();
			case 1:  return X_.y();
			case 2:  return X_.z();
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return iRow < 0 ? X_.x() : X_.z();
			}
		case 1:
			switch (iRow)
			{
			case 0:  return Y_.x();
			case 1:  return Y_.y();
			case 2:  return Y_.z();
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return iRow < 0 ? Y_.x() : Y_.z();
			}
		case 2:
			switch (iRow)
			{
			case 0:  return Z_.x();
			case 1:  return Z_.y();
			case 2:  return Z_.z();
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return iRow < 0 ? Z_.x() : Z_.z();
			}
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return iCol < 0 ? X_.x() : Z_.z();
		}
	}

private:
	column_type X_;
	column_type Y_;
	column_type Z_;
};


//
template<typename T>
T RMatrix3<T>::angle() const
{
	// solution 1
	const T s = (T)std::sqrt((X_.y() - Y_.x())*(X_.y() - Y_.x()) + (Y_.z() - Z_.y())*(Y_.z() - Z_.y()) + (Z_.x() - X_.z())*(Z_.x() - X_.z())) / T(2);
	// solution 2
	//const T s = -(T)std::sqrt((X_.y() - Y_.x())*(X_.y() - Y_.x()) + (Y_.z() - Z_.y())*(Y_.z() - Z_.y()) + (Z_.x() - X_.z())*(Z_.x() - X_.z())) / T(2);
	const T c = (X_.x() + Y_.y() + Z_.z() - T(1)) / T(2);
	if (MathUtil::isZero(c))
		return (T)std::asin(s) >= T(0) ? (T)MathConstant::PI_2 : (T)-MathConstant::PI_2;
	else return (T)std::atan2(s, c);
}

template<typename T>
Vector3<T> RMatrix3<T>::axis() const
{
	const T tAngle(angle());
	if (MathUtil::isZero(tAngle)) return Vector3<T>();

	Vector3<T> axis;
	if (MathUtil::isZero(tAngle + (T)MathConstant::PI) || MathUtil::isZero(tAngle - (T)MathConstant::PI))
	{
/*
		if (X_.x() + T(1) < T(0) || Y_.y() + T(1) < T(0) || Z_.z() + T(1) < T(0))
		{
			throw LogException(LogException::L_ERROR, "domian error", __FILE__, __LINE__, __FUNCTION__);
			//return RMatrix3<T>();
		}
*/
		// solution 1
		axis.x() = (T)std::sqrt((X_.x() + T(1)) / T(2));
		// solution 2
		//axis.x() = -(T)std::sqrt((X_.x() + T(1)) / T(2));
		axis.y() = (T)std::sqrt((Y_.y() + T(1)) / T(2));
		axis.z() = (T)std::sqrt((Z_.z() + T(1)) / T(2));

		if (axis.x() * Y_.x() < T(0)) axis.y() = -axis.y();
		if (axis.x() * Z_.x() < T(0)) axis.z() = -axis.z();
	}
	else
	{
		const T denom(T(2) * (T)std::sin(tAngle));
		axis.x() = (Y_.z() - Z_.y()) / denom;
		axis.y() = (Z_.x() - X_.z()) / denom;
		axis.z() = (X_.y() - Y_.x()) / denom;
	}

	return axis;
}

template<typename T>
/*static*/ RMatrix3<T> RMatrix3<T>::toRotationMatrix(const T &rad, const Vector3<T> &axis)
// R = [  e0  e3  e6  ]
//     [  e1  e4  e7  ]
//     [  e2  e5  e8  ]
{
	// for making a unit quaternion
	const T tNorm = (T)std::sqrt(axis.x()*axis.x() + axis.y()*axis.y() + axis.z()*axis.z());
	if (MathUtil::isZero(tNorm))
	{
		throw LogException(LogException::L_ERROR, "divide by zero", __FILE__, __LINE__, __FUNCTION__);
		//return RMatrix3<T>();
	}

	const T x(axis.x() / tNorm), y(axis.y() / tNorm), z(axis.z() / tNorm);
	const T s = (T)std::sin(rad);
	const T c = (T)std::cos(rad);
	const T v = T(1) - c;

	RMatrix3<T> rotMat;
	rotMat.X_.x() = x * x * v + c;
	rotMat.X_.y() = x * y * v + z * s;
	rotMat.X_.z() = x * z * v - y * s;

	rotMat.Y_.x() = x * y * v - z * s;
	rotMat.Y_.y() = y * y * v + c;
	rotMat.Y_.z() = y * z * v + x * s;

	rotMat.Z_.x() = x * z * v + y * s;
	rotMat.Z_.y() = y * z * v - x * s;
	rotMat.Z_.z() = z * z * v + c;

	if (!rotMat.isValid()) rotMat.orthonormalize();
	return rotMat;
}

// quaternion ==> 3x3 rotation matrix
template<typename T>
/*static*/ RMatrix3<T> RMatrix3<T>::toRotationMatrix(const Quaternion<T> &quat)
// R = [  e0  e3  e6  ]
//     [  e1  e4  e7  ]
//     [  e2  e5  e8  ]
{
	const Quaternion<T> unitQuat(quat.unit());

	RMatrix3<T> rotMat;
	rotMat.X_.x() = T(2) * (unitQuat.q0()*unitQuat.q0() + unitQuat.q1()*unitQuat.q1()) - T(1);
	//rotMat.X_.x() = T(1) - T(2) * (unitQuat.q2()*unitQuat.q2() + unitQuat.q3()*unitQuat.q3());
	rotMat.X_.y() = T(2) * (unitQuat.q1()*unitQuat.q2() + unitQuat.q0()*unitQuat.q3());
	rotMat.X_.z() = T(2) * (unitQuat.q1()*unitQuat.q3() - unitQuat.q0()*unitQuat.q2());

	rotMat.Y_.x() = T(2) * (unitQuat.q2()*unitQuat.q1() - unitQuat.q0()*unitQuat.q3());
	rotMat.Y_.y() = T(2) * (unitQuat.q0()*unitQuat.q0() + unitQuat.q2()*unitQuat.q2()) - T(1);
	//rotMat.Y_.y() = T(1) - T(2) * (unitQuat.q1()*unitQuat.q1() + unitQuat.q3()*unitQuat.q3());
	rotMat.Y_.z() = T(2) * (unitQuat.q2()*unitQuat.q3() + unitQuat.q0()*unitQuat.q1());

	rotMat.Z_.x() = T(2) * (unitQuat.q3()*unitQuat.q1() + unitQuat.q0()*unitQuat.q2());
	rotMat.Z_.y() = T(2) * (unitQuat.q3()*unitQuat.q2() - unitQuat.q0()*unitQuat.q1());
	rotMat.Z_.z() = T(2) * (unitQuat.q0()*unitQuat.q0() + unitQuat.q3()*unitQuat.q3()) - T(1);
	//rotMat.Z_.z() = T(1) - T(2) * (unitQuat.q1()*unitQuat.q1() + unitQuat.q2()*unitQuat.q2());

	if (!rotMat.isValid()) rotMat.orthonormalize();
	return rotMat;
}


//-----------------------------------------------------------------------------------------
// 3D Rotation Matrix API

template<typename T>
std::istream & operator>>(std::istream &stream, RMatrix3<T> &mat)
{
	// [> Xx  Yx  Zx  ] means a rotation matrix
	// [  Xy  Yy  Zy  ]
	// [  Xz  Yz  Zz <]
	char ch;
	stream >> ch >> ch >> mat.X().x() >> mat.Y().x() >> mat.Z().x() >> ch
		   >> ch >> mat.X().y() >> mat.Y().y() >> mat.Z().y() >> ch
		   >> ch >> mat.X().z() >> mat.Y().z() >> mat.Z().z() >> ch >> ch;
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream &stream, const RMatrix3<T> &mat)
{
	// [> Xx  Yx  Zx  ] means a rotation matrix
	// [  Xy  Yy  Zy  ]
	// [  Xz  Yz  Zz <]
	const bool bIsNewLineInserted = false;
	if (bIsNewLineInserted)
	{
		const int nWidth = 8;
		stream << "[> " << std::setw(nWidth) << mat.X().x() << std::setw(nWidth) << mat.Y().x() << std::setw(nWidth) << mat.Z().x() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().y() << std::setw(nWidth) << mat.Y().y() << std::setw(nWidth) << mat.Z().y() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().z() << std::setw(nWidth) << mat.Y().z() << std::setw(nWidth) << mat.Z().z() << " <]";
	}
	else
	{
		stream << "[> " << mat.X().x() << "  " << mat.Y().x() << "  " << mat.Z().x() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().y() << "  " << mat.Y().y() << "  " << mat.Z().y() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().z() << "  " << mat.Y().z() << "  " << mat.Z().z() << " <]";
	}
	return stream;
}

}  // namespace swl


#endif  // __SWL_MATH__ROTATION_MATRIX__H_
