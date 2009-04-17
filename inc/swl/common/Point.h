#if !defined(__SWL_COMMON__POINT__H_)
#define __SWL_COMMON__POINT__H_ 1


namespace swl {

//-----------------------------------------------------------------------------------------
// class Point2

template<typename T>
class Point2
{
public:
    typedef T value_type;

public:    
    Point2(const T& X = T(0), const T& Y = T(0)) : x(X), y(Y)  {}
    Point2(const Point2<T>& rhs) : x(rhs.x), y(rhs.y)  {}
	explicit Point2(const T rhs[2]) : x(rhs[0]), y(rhs[1])  {}
    ~Point2()  {}

    Point2<T>& operator=(const Point2<T>& rhs)
    {
        if (this == &rhs)  return *this;
        x = rhs.x;  y = rhs.y;
        return *this;
    }

public:
	///
    Point2<T>& operator+()  {  return *this;  }
    const Point2<T>& operator+() const  {  return *this;  }
    Point2<T> operator+(const Point2<T>& rhs) const
    {  return Point2<T>(x+rhs.x, y+rhs.y);  }
    Point2<T>& operator+=(const Point2<T>& rhs)
    {  x+=rhs.x;  y+=rhs.y;  return *this;  }
    Point2<T> operator-() const  {  return Point2<T>(-x, -y);  }
    Point2<T> operator-(const Point2<T>& rhs) const
    {  return Point2<T>(x-rhs.x, y-rhs.y);  }
    Point2<T>& operator-=(const Point2<T>& rhs)
    {  x-=rhs.x;  y-=rhs.y;  return *this;  }

	/// scalar operator
    Point2<T> operator*(const T& S) const
    {  return Point2<T>(x*S, y*S);  }
    Point2<T>& operator*=(const T& S)
    {  x*=S;  y*=S;  return *this;  }
    Point2<T> operator/(const T& S) const
    {  
		T Tol = T(1.0e-5);
		if (S >= -Tol && S <= Tol)  return *this;
		return Point2<T>(x/S, y/S);
	}
    Point2<T>& operator/=(const T& S)
    {
		T Tol = T(1.0e-5);
		if (S >= -Tol && S <= Tol)  return *this;
		x /= S;  y /= S;
		return *this;
	}

	///
	bool operator==(const Point2<T>& rhs) const  {  return IsEqual(rhs);  }
	bool operator!=(const Point2<T>& rhs) const  {  return !IsEqual(rhs);  }

protected:
	bool isEqual(const Point2<T>& rPt, T Tol = T(1.0e-5)) const
	{  return (x-rPt.x >= -Tol && x-rPt.x <= Tol) && (y-rPt.y >= -Tol && y-rPt.y <= Tol);  }

public:
	union
	{
		struct  {  T x, y;  };
		T point[2];
	};
};


//-----------------------------------------------------------------------------------------
// class Point3

template<typename T>
struct Point3
{
public:
    typedef T value_type;

public:    
    Point3(const T& X = T(0), const T& Y = T(0), const T& Z = T(0)) : x(X), y(Y), z(Z)  {}
    Point3(const Point3<T>& rhs) : x(rhs.x), y(rhs.y), z(rhs.z)  {}
	explicit Point3(const T rhs[3]) : x(rhs[0]), y(rhs[1]), z(rhs[2])  {}
    ~Point3()  {}

    Point3<T>& operator=(const Point3<T>& rhs)
    {
        if (this == &rhs)  return *this;
        x = rhs.x;  y = rhs.y;  z = rhs.z;
        return *this;
    }

public:
	///
    Point3<T>& operator+()  {  return *this;  }
    const Point3<T>& operator+() const  {  return *this;  }
    Point3<T> operator+(const Point3<T>& rhs) const
    {  return Point3<T>(x+rhs.x, y+rhs.y, z+rhs.z);  }
    Point3<T>& operator+=(const Point3<T>& rhs)
    {  x+=rhs.x;  y+=rhs.y;  z+=rhs.z;  return *this;  }
    Point3<T> operator-() const  {  return Point3<T>(-x, -y, -z);  }
    Point3<T> operator-(const Point3<T>& rhs) const
    {  return Point3<T>(x-rhs.x, y-rhs.y, z-rhs.z);  }
    Point3<T>& operator-=(const Point3<T>& rhs)
    {  x-=rhs.x;  y-=rhs.y;  z-=rhs.z;  return *this;  }

	/// scalar operator
    Point3<T> operator*(const T& S) const
    {  return Point3<T>(x*S, y*S, z*S);  }
    Point3<T>& operator*=(const T& S)
    {  x*=S;  y*=S;  z*=S;  return *this;  }
    Point3<T> operator/(const T& S) const
    {  
		T Tol = T(1.0e-5);
		if (S >= -Tol && S <= Tol)  return *this;
		return Point3<T>(x/S, y/S, z/S);
	}
    Point3<T>& operator/=(const T& S)
    {
		T Tol = T(1.0e-5);
		if (S >= -Tol && S <= Tol)  return *this;
		x /= S;  y /= S;  z /= S;
		return *this;
	}

	///
	bool operator==(const Point3<T>& rhs) const  {  return IsEqual(rhs);  }
	bool operator!=(const Point3<T>& rhs) const  {  return !IsEqual(rhs);  }

protected:
	bool isEqual(const Point3<T>& rPt, T Tol = T(1.0e-5)) const
	{  return (x-rPt.x >= -Tol && x-rPt.x <= Tol) && (y-rPt.y >= -Tol && y-rPt.y <= Tol) && (z-rPt.z >= -Tol && z-rPt.z <= Tol);  }

public:
	union
	{
		struct  {  T x, y, z;  };
		T point[3];
	};
};

}  // namespace swl


#endif  //  __SWL_COMMON__POINT__H_
