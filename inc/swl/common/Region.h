#if !defined(__SWL_COMMON__REGION__H_)
#define __SWL_COMMON__REGION__H_ 1


#include "swl/common/Point.h"


namespace swl {

//------------------------------------------------------------------------------------------
// class Region2: axis-aligned rectangle

template<typename T>
class Region2
{
public:
    typedef T value_type;

public:
    Region2(const T& x1 = T(0), const T& y1 = T(0), const T& x2 = T(0), const T& y2 = T(0))
    : left(x1 <= x2 ? x1 : x2), bottom(y1 <= y2 ? y1 : y2),
      right(x1 > x2 ? x1 : x2), top(y1 > y2 ? y1 : y2)
    {}
    Region2(const Point2<T>& pt1, const Point2<T>& pt2)
    : left(pt1.x <= pt2.x ? pt1.x : pt2.x), bottom(pt1.y <= pt2.y ? pt1.y : pt2.y),
      right(pt1.x > pt2.x ? pt1.x : pt2.x), top(pt1.y > pt2.y ? pt1.y : pt2.y)
    {}
    Region2(const Region2<T>& rhs)
    : left(rhs.left), bottom(rhs.bottom), right(rhs.right), top(rhs.top)  {}
    explicit Region2(const T rhs[4])
    : left(rhs[0] <= rhs[2] ? rhs[0] : rhs[2]), bottom(rhs[1] <= rhs[3] ? rhs[1] : rhs[3]),
      right(rhs[0] > rhs[2] ? rhs[0] : rhs[2]), top(rhs[1] > rhs[3] ? rhs[1] : rhs[3])
    ~Region2()  {}
    
    Region2<T>& operator=(const Region2<T>& rhs)
    {
        if (this == &rhs)  return *this;
        left = rhs.left;  bottom = rhs.bottom;
        right = rhs.right;  top = rhs.top;
        return *this;
    }

public:
    /// union rectangles               
    Region2<T> operator|(const Region2<T>& rhs) const
    {
        return Region2<T>(
            left < rhs.left ? left : rhs.left,
            bottom < rhs.bottom ? bottom : rhs.bottom,
            right >= rhs.right ? right : rhs.right,
            top >= rhs.top ? top : rhs.top
        );
    }
    Region2<T>& operator|=(const Region2<T>& rhs)
    {
        left = left < rhs.left ? left : rhs.left;
        bottom = bottom < rhs.bottom ? bottom : rhs.bottom;
        right = right >= rhs.right ? right : rhs.right;
        top = top >= rhs.top ? top : rhs.top;
        return *this;
    }
    /// intersect rectangles               
    Region2<T> operator&(const Region2<T>& rhs) const
    {
		if (isOverlapped(rhs))
		{
			return Region2<T>(
				left >= rhs.left ? left : rhs.left,
				bottom >= rhs.bottom ? bottom : rhs.bottom,
				right < rhs.right ? right : rhs.right,
				top < rhs.top ? top : rhs.top
			);
		}
		else  return Region2<T>();
    }
    Region2<T>& operator&=(const Region2<T>& rhs)
    {
 		if (isOverlapped(rhs))
		{
			left = left >= rhs.left ? left : rhs.left;
			bottom = bottom >= rhs.bottom ? bottom : rhs.bottom;
			right = right < rhs.right ? right : rhs.right;
			top = top < rhs.top ? top : rhs.top;
		}

		return *this;
    }

	///
	bool operator==(const Region2<T>& rhs) const  {  return isEqual(rhs);  }
	bool operator!=(const Region2<T>& rhs) const  {  return !isEqual(rhs);  }

	///
	Region2<T> operator+(const T& t) const
	{  return Region2<T>(left+t, bottom+t, right+t, top+t);  }
	Region2<T>& operator+=(const T& t)
    {  left+=t;  bottom+=t;  right+=t;  top+=t;  return *this;  }
	Region2<T> operator-(const T& t) const
	{  return Region2<T>(left-t, bottom-t, right-t, top-t);  }
	Region2<T>& operator-=(const T& t)
    {  left-=t;  bottom-=t;  right-=t;  top-=t;  return *this;  }
	Region2<T> operator*(const T& t) const
	{  return Region2<T>(left*t, bottom*t, right*t, top*t);  }
	Region2<T>& operator*=(const T& t)
    {  left*=t;  bottom*=t;  right*=t;  top*=t;  return *this;  }
    Region2<T> operator/(const T& t) const
    {  
		T Tol = T(1.0e-5);
		if (t >= -Tol && t <= Tol)  return *this;
		return Region2<T>(left/t, bottom/t, right/t, top/t);
	}
    Region2<T>& operator/=(const T& t)
    {
		T Tol = T(1.0e-5);
		if (t >= -Tol && t <= Tol)  return *this;
		left/=t;  bottom/=t;  right/=t;  top/=t;
		return *this;
	}

    /// move rectangle
    Region2<T> operator+(const Point2<T>& rPt) const
	{  return Region2<T>(left+rPt.x, bottom+rPt.y, right+rPt.x, top+rPt.y);  }
    Region2<T>& operator+=(const Point2<T>& rPt)
    {  left+=rPt.x;  bottom+=rPt.y;  right+=rPt.x;  top+=rPt.y;  return *this;  }
    Region2<T> operator-(const Point2<T>& rPt) const
	{  return Region2<T>(left-rPt.x, bottom-rPt.y, right-rPt.x, top-rPt.y);  }
    Region2<T>& operator-=(const Point2<T>& rPt)
    {  left-=rPt.x;  bottom-=rPt.y;  right-=rPt.x;  top-=rPt.y;  return *this;  }

    T getCenterX() const  {  return (left+right)/T(2);  }
    T getCenterY() const  {  return (bottom+top)/T(2);  }
    Point2<T> getCenter() const  {  return Point2<T>((left+right)/T(2), (bottom+top)/T(2));  }
    T getWidth() const  {  return right-left;  }
    T getHeight() const  {  return top-bottom;  }
    Point2<T> getSize() const  {  return Point2<T>(right-left, top-bottom);  }
	T getDiagonal() const  {  return (T)sqrt(getWidth()*getWidth() + getHeight()*getHeight());  }

	void moveCenter(const T& tDeltaX, const T& tDeltaY)
	{  left += tDeltaX;  bottom += tDeltaY;  right += tDeltaX;  top += tDeltaY;  }
	void changeSize(T tWidth, T tHeight, bool bIsCenterFixed = true)
	{
		if (bIsCenterFixed)
		{
			T tDeltaX = (tWidth - getWidth()) / T(2);
			T tDeltaY = (tHeight - getHeight()) / T(2);
			left -= tDeltaX;  bottom -= tDeltaY;
			right += tDeltaX;  top += tDeltaY;
		}
		else
		{
			right = left + tWidth;
			top = bottom + tHeight;
		}
	}
    
    /// inflate rectangle if dDelta is positive, deflate the one otherwise
    void inflate(const T& tDelta)
    {  left -= tDelta;  bottom -= tDelta;  right += tDelta;  top += tDelta;  }
    
    bool isIncluded(const T& tX, const T& tY, const T& Tol = T(1.0e-5)) const
    {
		return tX >= left-Tol && tX <= right+Tol
			   && tY >= bottom-Tol && tY <= top+Tol;
	}
    bool isIncluded(const Point2<T>& rPt, const T& Tol = T(1.0e-5)) const
    {
		return rPt.x >= left-Tol && rPt.x <= right+Tol
			   && rPt.y >= bottom-Tol && rPt.y <= top+Tol;
	}
    
    bool isIncluded(const Region2<T>& rRgn, const T& Tol = T(1.0e-5)) const
    {
		return isIncluded(rRgn.left, rRgn.bottom, Tol)
			   && isIncluded(rRgn.right, rRgn.top, Tol);
			   //&& isIncluded(rRgn.right, rRgn.bottom, Tol)
			   //&& isIncluded(rRgn.left, rRgn.top, Tol);
	}
    bool isOverlapped(const Region2<T>& rRgn, const T& Tol = T(1.0e-5)) const
    {
        T tLeft = left >= rRgn.left ? left : rRgn.left;
        T tBottom = bottom >= rRgn.bottom ? bottom : rRgn.bottom;
        T tRight = right < rRgn.right ? right : rRgn.right;
        T tTop = top < rRgn.top ? top : rRgn.top;

		return tLeft-Tol < tRight && tBottom-Tol < tTop;
/*
		return isIncluded(rRgn.left, rRgn.bottom, Tol)
			   || isIncluded(rRgn.right, rRgn.top, Tol)
			   || isIncluded(rRgn.right, rRgn.bottom, Tol)
			   || isIncluded(rRgn.left, rRgn.top, Tol)
			   || rRgn.isIncluded(*this, Tol);
*/
	}
    
    bool isValid() const  {  return (left < right) && (bottom < top);  }

protected:
	bool isEqual(const Region2<T>& rRgn, const T& Tol = T(1.0e-5)) const
	{
	    return (left-rRgn.left >= -Tol && left-rRgn.left <= Tol) && (right-rRgn.right >= -Tol && right-rRgn.right <= Tol)
	           && (bottom-rRgn.bottom >= -Tol && bottom-rRgn.bottom <= Tol) && (top-rRgn.top >= -Tol && top-rRgn.top <= Tol);
	}

public:
    union
	{
        struct  {  T left, bottom, right, top;  };
        T region[4];
    }; 
};


//------------------------------------------------------------------------------------------
// class Region3: axis-aligned box

template<typename T>
class Region3
{
public:
    typedef T value_type;

public:
    Region3(const T& x1 = T(0), const T& y1 = T(0), const T& z1 = T(0), const T& x2 = T(0), const T& y2 = T(0), const T& z2 = T(0))
    : left(x1 <= x2 ? x1 : x2), bottom(y1 <= y2 ? y1 : y2), front(z1 <= z2 ? z1 : z2),
      right(x1 > x2 ? x1 : x2), top(y1 > y2 ? y1 : y2), rear(z1 > z2 ? z1 : z2)
    {}
    Region3(const Point3<T>& pt1, const Point3<T>& pt2)
    : left(pt1.x <= pt2.x ? pt1.x : pt2.x), bottom(pt1.y <= pt2.y ? pt1.y : pt2.y), front(pt1.z <= pt2.z ? pt1.z : pt2.z),
      right(pt1.x > pt2.x ? pt1.x : pt2.x), top(pt1.y > pt2.y ? pt1.y : pt2.y), rear(pt1.z > pt2.z ? pt1.z : pt2.z)
    {}
    Region3( const Region3<T>& rhs)
    : left(rhs.left), bottom(rhs.bottom), front(rhs.front), right(rhs.right), top(rhs.top), rear(rhs.rear)
	{}
    explicit Region3( const T rhs[6])
    : left(rhs[0] <= rhs[3] ? rhs[0] : rhs[3]), bottom(rhs[1] <= rhs[4] ? rhs[1] : rhs[4]), front(rhs[2] <= rhs[5] ? rhs[2] : rhs[5]),
      right(rhs[0] > rhs[3] ? rhs[0] : rhs[3]), top(rhs[1] > rhs[4] ? rhs[1] : rhs[4]), rear(rhs[2] > rhs[5] ? rhs[2] : rhs[5])

    ~Region3()  {}
    
    Region3<T>& operator=(const Region3<T>& rhs)
    {
        if (this == &rhs)  return *this;
        left = rhs.left;  bottom = rhs.bottom;  front = rhs.front;
        right = rhs.right;  top = rhs.top;  rear = rhs.rear;
        return *this;
    }

public:
    /// union boxes               
    Region3<T> operator|(const Region3<T>& rhs) const
    {
        return Region3<T>(
            left < rhs.left ? left : rhs.left,
            bottom < rhs.bottom ? bottom : rhs.bottom,
            front < rhs.front ? front : rhs.front,
            right >= rhs.right ? right : rhs.right,
            top >= rhs.top ? top : rhs.top,
            rear >= rhs.rear ? rear : rhs.rear
        );
    }
    Region3<T>& operator|=(const Region3<T>& rhs)
    {
        left = left < rhs.left ? left : rhs.left;
        bottom = bottom < rhs.bottom ? bottom : rhs.bottom;
        front = front < rhs.front ? front : rhs.front;
        right = right >= rhs.right ? right : rhs.right;
        top = top >= rhs.top ? top : rhs.top;
        rear = rear >= rhs.rear ? rear : rhs.rear;
        return *this;
    }
    
    /// intersect boxes               
    Region3<T> operator&(const Region3<T>& rhs) const
    {
		if (isOverlapped(rhs))
		{
			return Region3<T>(
				left >= rhs.left ? left : rhs.left,
				bottom >= rhs.bottom ? bottom : rhs.bottom,
				front >= rhs.front ? front : rhs.front,
				right < rhs.right ? right : rhs.right,
				top < rhs.top ? top : rhs.top,
				rear < rhs.rear ? rear : rhs.rear
			);
		}
		else  return Region3<T>();
    }
    Region3<T>& operator&=(const Region3<T>& rhs)
    {
		if (isOverlapped(rhs))
		{
			left = left >= rhs.left ? left : rhs.left;
			bottom = bottom >= rhs.bottom ? bottom : rhs.bottom;
			front = front >= rhs.front ? front : rhs.front;
			right = right < rhs.right ? right : rhs.right;
			top = top < rhs.top ? top : rhs.top;
			rear = rear < rhs.rear ? rear : rhs.rear;
		}

        return *this;
    }

	///
	bool operator==(const Region3<T>& rhs) const  {  return isEqual(rhs);  }
	bool operator!=(const Region3<T>& rhs) const  {  return !isEqual(rhs);  }

	///
	Region3<T> operator+(const T& t) const
	{  return Region2<T>(left+t, bottom+t, front+t, right+t, top+t, rear+t);  }
	Region3<T>& operator+=(const T& t)
    {  left+=t;  bottom+=t;  front+=t;  right+=t;  top+=t;  rear+=t;  return *this;  }
	Region3<T> operator-(const T& t) const
	{  return Region2<T>(left-t, bottom-t, front-t, right-t, top-t, rear-t);  }
	Region3<T>& operator-=(const T& t)
    {  left-=t;  bottom-=t;  front-=t;  right-=t;  top-=t;  rear-=t;  return *this;  }
	Region3<T> operator*(const T& t) const
	{  return Region2<T>(left*t, bottom*t, front*t, right*t, top*t, rear*t);  }
	Region3<T>& operator*=(const T& t)
    {  left*=t;  bottom*=t;  front*=t;  right*=t;  top*=t;  rear*=t;  return *this;  }
    Region3<T> operator/(const T& t) const
    {  
		T Tol = T(1.0e-5);
		if (t >= -Tol && t <= Tol)  return *this;
		return Region3<T>(left/t, bottom/t, front/t, right/t, top/t, rear/t);
	}
    Region3<T>& operator/=(const T& t)
    {
		T Tol = T(1.0e-5);
		if (t >= -Tol && t <= Tol)  return *this;
		left/=t;  bottom/=t;  front/=t;  right/=t;  top/=t;  rear/=t;
		return *this;
	}

    /// move box
    Region3<T> operator+(const Point3<T>& rPt) const
	{  return Region3<T>(left+rPt.x, bottom+rPt.y, front+rPt.z, right+rPt.x, top+rPt.y, rear+rPt.z);  }
    Region3<T>& operator+=(const Point3<T>& rPt)
    {  left+=rPt.x;  bottom+=rPt.y;  front+=rPt.z;  right+=rPt.x;  top+=rPt.y;  rear+=rPt.z;  return *this;  }
    Region3<T> operator-(const Point3<T>& rPt) const
	{  return Region3<T>(left-rPt.x, bottom-rPt.y, front-rPt.z, right-rPt.x, top-rPt.y, rear-rPt.z);  }
    Region3<T>& operator-=(const Point3<T>& rPt)
    {  left-=rPt.x;  bottom-=rPt.y;  front-=rPt.z;  right-=rPt.x;  top-=rPt.y;  rear-=rPt.z;  return *this;  }

    ///
    T getCenterX() const  {  return (left+right)/T(2);  }
    T getCenterY() const  {  return (bottom+top)/T(2);  }
    T getCenterZ() const  {  return (front+rear)/T(2);  }
    Point3<T> getCenter() const  {  return Point3<T>((left+right)/T(2), (bottom+top)/T(2), (front+rear)/T(2));  }
    T getWidth() const  {  return right-left;  }
    T getHeight() const  {  return top-bottom;  }
    T Depth() const  {  return rear-front;  }
    Point3<T> getSize() const  {  return Point3<T>(right-left, top-bottom, rear-front);  }
	T getDiagonal() const  {  return (T)sqrt(getWidth()*getWidth() + getHeight()*getHeight() + Depth()*Depth());  }

    ///
	void moveCenter(const T& tDeltaX, const T& tDeltaY, const T& tDeltaZ)
	{  left += tDeltaX;  bottom += tDeltaY;  front += tDeltaZ;  right += tDeltaX;  top += tDeltaY;  rear += tDeltaZ;  }
	void changeSize(const T& tWidth, const T& tHeight, const T& tDepth, bool bIsCenterFixed = true)
	{
		if (bIsCenterFixed)
		{
			T tDeltaX = (tWidth - getWidth()) / T(2);
			T tDeltaY = (tHeight - getHeight()) / T(2);
			T tDeltaZ = (tDepth - Depth()) / T(2);
			left -= tDeltaX;  bottom -= tDeltaY;  front -= tDeltaZ;
			right += tDeltaX;  top += tDeltaY;  rear += tDeltaZ;
		}
		else
		{
			right = left + tWidth;
			top = bottom + tHeight;
			rear = front + tDepth;
		}
	}
    
    /// inflate box if dDelta is positive, deflate the one otherwise
    void inflate(const T& tDelta)
    {  left -= tDelta;  bottom -= tDelta;  front -= tDelta;  right += tDelta;  top += tDelta;  rear += tDelta;  }
    
    bool isIncluded(const T& tX, const T& tY, const T& tZ, const T& Tol = T(1.0e-5)) const
    {
		return tX >= left-Tol && tX <= right+Tol
			   && tY >= bottom-Tol && tY <= top+Tol
			   && tZ >= front-Tol && tZ <= rear+Tol;
	}
    bool isIncluded(const Point3<T>& rPt, const T& Tol = T(1.0e-5)) const
    {
		return rPt.x >= left-Tol && rPt.x <= right+Tol
			   && rPt.y >= bottom-Tol && rPt.y <= top+Tol
			   && rPt.z >= front-Tol && rPt.z <= rear+Tol;
	}
    
    bool isIncluded(const Region3<T>& rRgn, const T& Tol = T(1.0e-5)) const
    {
		return isIncluded(rRgn.left, rRgn.bottom, rRgn.front, Tol)
			   && isIncluded(rRgn.right, rRgn.top, rRgn.rear, Tol);
	}
    bool isOverlapped(const Region3<T>& rRgn, const T& Tol = T(1.0e-5)) const
    {
		T tLeft = left >= rRgn.left ? left : rRgn.left;
		T tBottom = bottom >= rRgn.bottom ? bottom : rRgn.bottom;
		T tFront = front >= rRgn.front ? front : rRgn.front;
		T tRight = right < rRgn.right ? right : rRgn.right;
		T tTop = top < rRgn.top ? top : rRgn.top;
		T tRear = rear < rRgn.rear ? rear : rRgn.rear;

		return tLeft-Tol < tRight && tBottom-Tol < tTop && tFront-Tol < tRear ? true : false;
/*
		return isIncluded(rRgn.left, rRgn.bottom, rRgn.front, Tol)
			   || isIncluded(rRgn.left, rRgn.bottom, rRgn.rear, Tol);
			   || isIncluded(rRgn.left, rRgn.top, rRgn.front, Tol);
			   || isIncluded(rRgn.left, rRgn.top, rRgn.rear, Tol);
			   || isIncluded(rRgn.right, rRgn.bottom, rRgn.front, Tol);
			   || isIncluded(rRgn.right, rRgn.bottom, rRgn.rear, Tol);
			   || isIncluded(rRgn.right, rRgn.top, rRgn.front, Tol);
			   || isIncluded(rRgn.right, rRgn.top, rRgn.rear, Tol);
			   || rRgn.isIncluded(*this);
*/
	}
    
    bool isValid() const  {  return (left < right) && (bottom < top) && (front < rear);  }

protected:
	bool isEqual(const Region3<T>& rRgn, const T& Tol = T(1.0e-5)) const
	{
	    return (left-rRgn.left >= -Tol && left-rRgn.left <= Tol) && (right-rRgn.right >= -Tol && right-rRgn.right <= Tol)
	           && (bottom-rRgn.bottom >= -Tol && bottom-rRgn.bottom <= Tol) && (top-rRgn.top >= -Tol && top-rRgn.top <= Tol)
	           && (front-rRgn.front >= -Tol && front-rRgn.front <= Tol) && (rear-rRgn.rear >= -Tol && rear-rRgn.rear <= Tol);
    }

public:
    union
	{
        struct  {  T left, bottom, front, right, top, rear;  };
        T region[6];
    };
};

}  // namespace swl


#endif  //  __SWL_COMMON__REGION__H_
