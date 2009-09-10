#if !defined(__SWL_GRAPHICS__BOUNDING_BOX__H_)
#define __SWL_GRAPHICS__BOUNDING_BOX__H_ 1


#include "swl/math/TMatrix.h"
#include "swl/base/Region.h"
#include <cmath>


namespace swl {

//-----------------------------------------------------------------------------------------
// class BoundingBox: axis-aligned bounding box

template <typename T>
class BoundingBox
{
public:
	//typedef BoundingBox		base_type;
	typedef T					value_type;

public:
	BoundingBox()
	: //base_type(),
	  bound_()
	{}
	explicit BoundingBox(const T lowerArr[3], const T upperArr[3])
	: //base_type(),
	  bound_(lowerArr, upperArr)
	{}
	explicit BoundingBox(const T boundArr[6])
	: swObject(),
	  bound_(boundArr)
	{}
	BoundingBox(const BoundingBox<T>& rhs)
	: //base_type(rhs),
	  bound_(rhs.bound_)
	{}
	~BoundingBox()  {}

	BoundingBox<T>& operator=(const BoundingBox<T>& rhs)
	{
		if (this == &rhs) return *this;
		//static_cast<base_type &>(*this) = rhs;
		bound_ = rhs.bound_;
		return *this;
	}

public:
    /// union bounding boxes             
	BoundingBox<T> operator|(const BoundingBox<T>& rhs) const
	{  return bound_ | rhs.bound_;  }
	BoundingBox<T>& operator|=(const BoundingBox<T>& rhs)
	{  return (bound_ |= rhs.bound_);  }
	
    /// intersect bounding boxes               
	BoundingBox<T> operator&(const BoundingBox<T>& rhs) const
	{  return bound_ & rhs.bound_;  }
	BoundingBox<T>& operator&=(const BoundingBox<T>& rhs)
	{  return (bound_ &= rhs.bound_);  }

	///
	void set(const T lower[3], const T upper[3])
	{  bound_ = Region3<T>(lower, upper);  }
	void get(T lower[3], T upper[3])
	{
		for (int i=0 ; i<3 ; ++i)
		{
			lower[i] = bound_.lower[i];
			upper[i] = bound_.upper[i];
		}
	}

	///
	void updateBound(const TMatrix3<T>& rTMat)
	{  *this = calcAxisAlignedBox(rTMat);  }

	///
	void center(T centerArr[3])
	{
		centerArr[0] = bound_.centerX();
		centerArr[1] = bound_.centerY();
		centerArr[2] = bound_.centerZ();
	}
	T diagonal()
	{  return bound_.diagonal();  }

protected:
	BoundingBox<T> calcAxisAlignedBox(const TMatrix3<T>& mat)
	{
		swTVector3<T> vertexArr[8];
		getAllVertices(vertexArr);
		for (int i = 0 ; i < 8 ; ++i) vertexArr[i] = mat * vertexArr[i];

		T minArr[3], maxArr[3];
		for (int i = 0 ; i < 3 ; ++i) minArr[i] = maxArr[i] = vertexArr[0][i];

		for (int i = 1 ; i < 8 ; ++i)
			for (int j = 0 ; j < 3 ; ++j)
			{
				if (minArr[j] > vertexArr[i][j]) minArr[j] = vertexArr[i][j];
				else if (maxArr[j] < vertexArr[i][j]) maxArr[j] = vertexArr[i][j];
			}

		return BoundingBox<T>(minArr, maxArr);
	}

	void getAllVertices(swTVector3<T> vertexArr[8])
	{
		vertexArr[0].x() = bound_.lower[0];  vertexArr[0].y() = bound_.lower[1];  vertexArr[0].z() = bound_.lower[2];
		vertexArr[1].x() = bound_.lower[0];  vertexArr[1].y() = bound_.upper[1];  vertexArr[1].z() = bound_.lower[2];
		vertexArr[2].x() = bound_.upper[0];  vertexArr[2].y() = bound_.upper[1];  vertexArr[2].z() = bound_.lower[2];
		vertexArr[3].x() = bound_.upper[0];  vertexArr[3].y() = bound_.lower[1];  vertexArr[3].z() = bound_.lower[2];
		vertexArr[4].x() = bound_.lower[0];  vertexArr[4].y() = bound_.lower[1];  vertexArr[4].z() = bound_.upper[2];
		vertexArr[5].x() = bound_.lower[0];  vertexArr[5].y() = bound_.upper[1];  vertexArr[5].z() = bound_.upper[2];
		vertexArr[6].x() = bound_.upper[0];  vertexArr[6].y() = bound_.upper[1];  vertexArr[6].z() = bound_.upper[2];
		vertexArr[7].x() = bound_.upper[0];  vertexArr[7].y() = bound_.lower[1];  vertexArr[7].z() = bound_.upper[2];
	}

private:
	/// axis-aligned bounding box
	Region3<T> bound_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__BOUNDING_BOX__H_
