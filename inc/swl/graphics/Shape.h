#if !defined(__SWL_GRAPHICS__SHAPE__H_)
#define __SWL_GRAPHICS__SHAPE__H_ 1


#include "swl/graphics/GraphicsObj.h"
#include "swl/graphics/Geometry.h"
#include "swl/graphics/Appearance.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Shape

class SWL_GRAPHICS_API Shape: public GraphicsObj
{
public:
	typedef GraphicsObj base_type;

public:
	Shape();
	Shape(const Shape &rhs);
	virtual ~Shape();

	Shape & operator=(const Shape &rhs);

private:
	///
	Geometry geometry_;
	Appearance appearance_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SHAPE__H_
