#if !defined(__SWL_GRAPHICS__COORDINATE_FRAME__H_)
#define __SWL_GRAPHICS__COORDINATE_FRAME__H_ 1


#include "swl/graphics/GraphicsObj.h"
#include "swl/math/TMatrix.h"


namespace swl {

//-----------------------------------------------------------------------------------------
//

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_GRAPHICS_TEMPLATE_EXTERN template class SWL_GRAPHICS_API Vector3<double>;
SWL_GRAPHICS_TEMPLATE_EXTERN template class SWL_GRAPHICS_API TMatrix3<double>;
#endif


//-----------------------------------------------------------------------------------------
// class CoordinateFrame

class SWL_GRAPHICS_API CoordinateFrame: public GraphicsObj
{
public:
	typedef GraphicsObj			base_type;
	typedef TMatrix3<double>	frame_type;

public:
	CoordinateFrame(const bool isPrintable, const bool isPickable);
	CoordinateFrame(const CoordinateFrame &rhs);
	virtual ~CoordinateFrame();

	CoordinateFrame& operator=(const CoordinateFrame &rhs);

public:
	void setFrame(const frame_type &frame)  {  frame_ = frame;  }
	frame_type getFrame() const  {  return frame_;  }

private:
	//
	frame_type frame_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__COORDINATE_FRAME__H_
