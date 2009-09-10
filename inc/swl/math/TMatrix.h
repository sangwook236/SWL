#if !defined(__SWL_MATH__TRANSFORMATION_MATRIX__H_)
#define __SWL_MATH__TRANSFORMATION_MATRIX__H_ 1


#include "swl/math/TVector.h"
#include "swl/math/MathConstant.h"
#include "swl/math/MathUtil.h"
#include <iomanip>


namespace swl {

template<typename TT> class TVector2;
template<typename TT> class TVector3;


//----------------------------------------------------------------------------------------------
// class TMatrix2
// : 3x3 homogeneous transformation matrix for 2D homogeneous vector ( 3x1 vector )
// [  Xx  Yx  Tx  ]  =  [  X  Y  T  ]  =  [  R  T  ]  =  [  e0  e3  e6  ]
// [  Xy  Yy  Ty  ]     [  0  0  1  ]     [  0  1  ]     [  e1  e4  e7  ]
// [   0   0   1  ]								      [  e2  e5  e8  ]

template<typename TT>
class TMatrix2
{
public:
    typedef TT					value_type;
	typedef TVector2<TT>		column_type;

public:
	TMatrix2()
	: X_(TT(1), TT(0)), Y_(TT(0), TT(1)), T_(TT(0), TT(0))
	{}
	explicit TMatrix2(const column_type& rX, const column_type& rY, const column_type& rT)
	: X_(rX), Y_(rY), T_(rT)
	{}
	explicit TMatrix2(const TT rhs[9])
	: X_(rhs[0], rhs[1]), Y_(rhs[3], rhs[4]), T_(rhs[6], rhs[7])
	{}
	TMatrix2(const TMatrix2& rhs)
	: X_(rhs.X_), Y_(rhs.Y_), T_(rhs.T_)
	{}
	~TMatrix2()  {}

	TMatrix2& operator=(const TMatrix2& rhs)
	{
		if (this == &rhs) return *this;
		X_ = rhs.X_;  Y_ = rhs.Y_;  T_ = rhs.T_;
		return *this;
	}

public:
	///
	column_type& X()  {  return X_;  }
	const column_type& X() const  {  return X_;  }
	column_type& Y()  {  return Y_;  }
	const column_type& Y() const  {  return Y_;  }
	column_type& T()  {  return T_;  }
	const column_type& T() const  {  return T_;  }

	///
	TT operator[](int iIndex) const  {  return getEntry(iIndex%3, iIndex/3);  }
	TT operator()(int row, int col) const  {  return getEntry(row, col);  }
	
	///
	bool get(TT entry[9]) const 
	{
		entry[0] = X_.x();  entry[1] = X_.y();  entry[2] = TT(0);
		entry[3] = Y_.x();  entry[4] = Y_.y();  entry[5] = TT(0);
		entry[6] = T_.x();  entry[7] = T_.y();  entry[8] = TT(1);
		return true;
	}
	bool set(const TT entry[9])
	{
		X_.x() = entry[0];  X_.y() = entry[1];
		Y_.x() = entry[3];  Y_.y() = entry[4];
		T_.x() = entry[6];  T_.y() = entry[7];
		return isValid();
	}

	///
	bool isValid(const TT& tTol = (TT)MathConstant::EPS) const
	{  return X_.isUnit(tTol) && Y_.isUnit(tTol) && X_.isOrthogonal(Y_, tTol);  }
	bool isEqual(const TMatrix2& rhs, const TT& tTol = (TT)MathConstant::EPS) const
	{
		return X_.isEqual(rhs.X_, tTol) && Y_.isEqual(rhs.Y_, tTol) && T_.isEqual(rhs.T_, tTol);
	}

	/// comparison operator
    bool operator==(const TMatrix2& rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const TMatrix2& rhs) const  {  return !isEqual(rhs);  }

	///
	TMatrix2 operator*(const TMatrix2& rhs) const
	{
		return TMatrix2(
			X_*rhs.X_.x() + Y_*rhs.X_.y(),
			X_*rhs.Y_.x() + Y_*rhs.Y_.y(),
			X_*rhs.T_.x() + Y_*rhs.T_.y() + T_
		);
	}
	TMatrix2& operator*=(const TMatrix2& rhs)
	{  return *this = *this * rhs;  }
	column_type operator*(const column_type& rV) const
	{  return column_type(X_*rV.x() + Y_*rV.y() + T_);  }

	///
	void identity()
	{
		X_.x() = TT(1);  X_.y() = TT(0);
		Y_.x() = TT(0);  Y_.y() = TT(1);
		T_.x() = TT(0);  T_.y() = TT(0);
	}
	TMatrix2 inverse() const
	{
		return TMatrix2(
			column_type(X_.x(), Y_.x()), 
			column_type(X_.y(), Y_.y()), 
			column_type(-X_*T_, -Y_*T_)
		);
	}

	///
	bool orthonormalize()
	{
		if (X_.isZero() || Y_.isZero()) return false;
		Y_ = column_type(-X_.y(), X_.x());
		X_ = X_.unit();
		Y_ = Y_.unit();
		return true;
	}

protected:
	TT getEntry(int row, int col) const
	{
		switch (col)
		{
		case 0:
			switch (row)
			{
			case 0:  return X_.x();
			case 1:  return X_.y();
			case 2:  return TT(0);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? X_.x() : TT(0);
			}
		case 1:
			switch (row)
			{
			case 0:  return Y_.x();
			case 1:  return Y_.y();
			case 2:  return TT(0);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? Y_.x() : TT(0);
			}
		case 2:
			switch (row)
			{
			case 0:  return T_.x();
			case 1:  return T_.y();
			case 2:  return TT(1);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? T_.x() : TT(1);
			}
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return col < 0 ? X_.x() : TT(1);
		}
	}

private:
	column_type X_, Y_, T_;
};


//-----------------------------------------------------------------------------------------
// 2D Transformation Matrix API

template<typename T>
std::istream& operator>>(std::istream& stream, TMatrix2<T>& mat)
{
	// [> Xx  Yx  Tx  ] means a transformation matrix
	// [  Xy  Yy  Ty  ]
	// [   0   0   1 <]
	char ch;
	T t1, t2, t3;
	stream >> ch >> ch >> mat.X().x() >> mat.Y().x() >> mat.T().x() >> ch
		   >> ch >> mat.X().y() >> mat.Y().y() >> mat.T().y() >> ch
		   >> ch >> t1 >> t2 >> t3 >> ch >> ch;

	if (!MathUtil::isZero(t1) || !MathUtil::isZero(t2) || !MathUtil::isZero(t3 - T(1)))
		throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

	return stream;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const TMatrix2<T>& mat)
{
	// [> Xx  Yx  Tx  ] means a transformation matrix
	// [  Xy  Yy  Ty  ]
	// [   0   0   1 <]
	bool bIsNewLineInserted = false;
	if (bIsNewLineInserted)
	{
		int nWidth = 8;
		stream << "[> " << std::setw(nWidth) << mat.X().x() << std::setw(nWidth) << mat.Y().x() << std::setw(nWidth) << mat.T().x() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().y() << std::setw(nWidth) << mat.Y().y() << std::setw(nWidth) << mat.T().y() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) <<         '0' << std::setw(nWidth) <<         '0' << std::setw(nWidth) <<         '1' << " <]";
	}
	else
	{
		stream << "[> " << mat.X().x() << "  " << mat.Y().x() << "  " << mat.T().x() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().y() << "  " << mat.Y().y() << "  " << mat.T().y() << "  ]";
		stream << "  ";
		stream << "[  " <<         '0' << "  " <<         '0' << "  " <<         '1' << " <]";
	}
	return stream;
}


//----------------------------------------------------------------------------------------------
// class TMatrix3
// : 4x4 homogeneous transformation matrix for 3D homogeneous vector ( 4x1 vector )
// [  Xx  Yx  Zx  Tx  ]  =  [  X  Y  Z  T  ]  =  [  R  T  ]  =  [  e00  e04  e08  e12  ]
// [  Xy  Yy  Zy  Ty  ]     [  0  0  0  1  ]     [  0  1  ]     [  e01  e05  e09  e13  ]
// [  Xz  Yz  Zz  Tz  ]										    [  e02  e06  e10  e14  ]
// [   0   0   0   1  ]										    [  e03  e07  e11  e15  ]

template<typename TT>
class TMatrix3
{
public:
    typedef TT					value_type;
	typedef TVector3<TT>		column_type;

public:
	TMatrix3()
	: X_(TT(1), TT(0), TT(0)), Y_(TT(0), TT(1), TT(0)), Z_(TT(0), TT(0), TT(1)), T_(TT(0), TT(0), TT(0))
	{}
	explicit TMatrix3(const column_type& rX, const column_type& rY, const column_type& rZ, const column_type& rT)
	: X_(rX), Y_(rY), Z_(rZ), T_(rT)
	{}
	explicit TMatrix3(const TT rhs[16])
	: X_(rhs[0], rhs[1], rhs[2]), Y_(rhs[4], rhs[5], rhs[6]), Z_(rhs[8], rhs[9], rhs[10]), T_(rhs[12], rhs[13], rhs[14])
	{}
	TMatrix3(const TMatrix3& rhs)
	: X_(rhs.X_), Y_(rhs.Y_), Z_(rhs.Z_), T_(rhs.T_)
	{}
	~TMatrix3()  {}

	TMatrix3& operator=(const TMatrix3& rhs)
	{
		if (this == &rhs) return *this;
		X_ = rhs.X_;  Y_ = rhs.Y_;  Z_ = rhs.Z_;  T_ = rhs.T_;
		return *this;
	}

public:
	///
	column_type& X()  {  return X_;  }
	const column_type& X() const  {  return X_;  }
	column_type& Y()  {  return Y_;  }
	const column_type& Y() const  {  return Y_;  }
	column_type& Z()  {  return Z_;  }
	const column_type& Z() const  {  return Z_;  }
	column_type& T()  {  return T_;  }
	const column_type& T() const  {  return T_;  }

    ///
	TT operator[](int iIndex) const  {  return getEntry(iIndex%4, iIndex/4);  }
	TT operator()(int row, int col) const  {  return getEntry(row, col);  }
	
	///
	bool set(const TT entry[16])
	{
		X_.x() = entry[0];  X_.y() = entry[1];  X_.z() = entry[2];
		Y_.x() = entry[4];  Y_.y() = entry[5];  Y_.z() = entry[6];
		Z_.x() = entry[8];  Z_.y() = entry[9];  Z_.z() = entry[10];
		T_.x() = entry[12];  T_.y() = entry[13];  T_.z() = entry[14];
		return isValid();
	}
	bool get(TT entry[16]) const 
	{
		entry[0] = X_.x();  entry[1] = X_.y();  entry[2] = X_.z();  entry[3] = TT(0);
		entry[4] = Y_.x();  entry[5] = Y_.y();  entry[6] = Y_.z();  entry[7] = TT(0);
		entry[8] = Z_.x();  entry[9] = Z_.y();  entry[10] = Z_.z();  entry[11] = TT(0);
		entry[12] = T_.x();  entry[13] = T_.y();  entry[14] = T_.z();  entry[15] = TT(1);
		return true;
	}

	///
	bool isValid(const TT& tTol = (TT)MathConstant::EPS) const
	{
		return X_.isUnit(tTol) && Y_.isUnit(tTol) && Z_.isUnit(tTol) &&
			   X_.isOrthogonal(Y_, tTol) && X_.isOrthogonal(Z_, tTol) && Y_.isOrthogonal(Z_, tTol);
	}
	bool isEqual(const TMatrix3& rhs, const TT& tTol = (TT)MathConstant::EPS) const
	{
		return X_.isEqual(rhs.X_, tTol) && Y_.isEqual(rhs.Y_, tTol) &&
			   Z_.isEqual(rhs.Z_, tTol) && T_.isEqual(rhs.T_, tTol);
	}

	/// comparison operator
    bool operator==(const TMatrix3& rhs) const  {  return isEqual(rhs);  }
    bool operator!=(const TMatrix3& rhs) const  {  return !isEqual(rhs);  }

	///
	TMatrix3 operator*(const TMatrix3& rhs) const
	{
		return TMatrix3(
			X_*rhs.X_.x() + Y_*rhs.X_.y() + Z_*rhs.X_.z(),
			X_*rhs.Y_.x() + Y_*rhs.Y_.y() + Z_*rhs.Y_.z(),
			X_*rhs.Z_.x() + Y_*rhs.Z_.y() + Z_*rhs.Z_.z(),
			X_*rhs.T_.x() + Y_*rhs.T_.y() + Z_*rhs.T_.z() + T_
		);
	}
	TMatrix3& operator*=(const TMatrix3& rhs)
	{  return *this = *this * rhs;  }
	column_type operator*(const column_type& rV) const
	{  return column_type(X_*rV.x() + Y_*rV.y() + Z_*rV.z() + T_);  }

	///
	void identity()
	{
		X_.x() = TT(1);  X_.y() = TT(0);  X_.z() = TT(0);
		Y_.x() = TT(0);  Y_.y() = TT(1);  Y_.z() = TT(0);
		Z_.x() = TT(0);  Z_.y() = TT(0);  Z_.z() = TT(1);
		T_.x() = TT(0);  T_.y() = TT(0);  T_.z() = TT(0);
	}
	TMatrix3 inverse() const
	{
		return TMatrix3(
			column_type(X_.x(), Y_.x(), Z_.x()), 
			column_type(X_.y(), Y_.y(), Z_.y()), 
			column_type(X_.z(), Y_.z(), Z_.z()), 
			column_type(-X_*T_, -Y_*T_, -Z_*T_)
		);
	}

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

protected:
	TT getEntry(int row, int col) const
	{
		switch (col)
		{
		case 0:
			switch (row)
			{
			case 0:  return X_.x();
			case 1:  return X_.y();
			case 2:  return X_.z();
			case 3:  return TT(0);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? X_.x() : TT(0);
			}
		case 1:
			switch (row)
			{
			case 0:  return Y_.x();
			case 1:  return Y_.y();
			case 2:  return Y_.z();
			case 3:  return TT(0);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? Y_.x() : TT(0);
			}
		case 2:
			switch (row)
			{
			case 0:  return Z_.x();
			case 1:  return Z_.y();
			case 2:  return Z_.z();
			case 3:  return TT(0);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? Z_.x() : TT(0);
			}
		case 3:
			switch (row)
			{
			case 0:  return T_.x();
			case 1:  return T_.y();
			case 2:  return T_.z();
			case 3:  return TT(1);
			default:
				throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
				//return row < 0 ? T_.x() : TT(1);
			}
		default:
			throw LogException(LogException::L_ERROR, "illegal index", __FILE__, __LINE__, __FUNCTION__);
			//return col < 0 ? X_.x() : TT(1);
		}
	}

private:
	column_type X_, Y_, Z_, T_;
};


//-----------------------------------------------------------------------------------------
// 3D Transformation Matrix API

template<typename T>
std::istream& operator>>(std::istream& stream, TMatrix3<T>& mat)
{
	// [> Xx  Yx  Zx  Tx  ] means a transformation matrix
	// [  Xy  Yy  Zy  Ty  ]
	// [  Xz  Yz  Zz  Tz  ]
	// [   0   0   0   1 <]
	char ch;
	T t1, t2, t3, t4;
	stream >> ch >> ch >> mat.X().x() >> mat.Y().x() >> mat.Z().x() >> mat.T().x() >> ch
		   >> ch >> mat.X().y() >> mat.Y().y() >> mat.Z().y() >> mat.T().y() >> ch
		   >> ch >> mat.X().z() >> mat.Y().z() >> mat.Z().z() >> mat.T().z() >> ch
		   >> ch >> t1 >> t2 >> t3 >> t4 >> ch >> ch;

	if (!MathUtil::isZero(t1) || !MathUtil::isZero(t2) || !MathUtil::isZero(t3) || !MathUtil::isZero(t4 - T(1)))
		throw LogException(LogException::L_ERROR, "illegal value", __FILE__, __LINE__, __FUNCTION__);

	return stream;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const TMatrix3<T>& mat)
{
	// [> Xx  Yx  Zx  Tx  ] means a transformation matrix
	// [  Xy  Yy  Zy  Ty  ]
	// [  Xz  Yz  Zz  Tz  ]
	// [   0   0   0   1 <]
	const bool bIsNewLineInserted = false;
	if (bIsNewLineInserted)
	{
		int nWidth = 8;
		stream << "[> " << std::setw(nWidth) << mat.X().x() << std::setw(nWidth) << mat.Y().x() << std::setw(nWidth) << mat.Z().x() << std::setw(nWidth) << mat.T().x() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().y() << std::setw(nWidth) << mat.Y().y() << std::setw(nWidth) << mat.Z().y() << std::setw(nWidth) << mat.T().y() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) << mat.X().z() << std::setw(nWidth) << mat.Y().z() << std::setw(nWidth) << mat.Z().z() << std::setw(nWidth) << mat.T().z() << "  ]";
		stream << '\n';
		stream << "[  " << std::setw(nWidth) <<         '0' << std::setw(nWidth) <<         '0' << std::setw(nWidth) <<         '0' << std::setw(nWidth) <<         '1' << " <]";
	}
	else
	{
		stream << "[> " << mat.X().x() << "  " << mat.Y().x() << "  " << mat.Z().x() << "  " << mat.T().x() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().y() << "  " << mat.Y().y() << "  " << mat.Z().y() << "  " << mat.T().y() << "  ]";
		stream << "  ";
		stream << "[  " << mat.X().z() << "  " << mat.Y().z() << "  " << mat.Z().z() << "  " << mat.T().z() << "  ]";
		stream << "  ";
		stream << "[  " <<         '0' << "  " <<         '0' << "  " <<         '0' << "  " <<         '1' << " <]";
	}
	return stream;
}

}  // namespace swl


#endif  // __SWL_MATH__TRANSFORMATION_MATRIX__H_
