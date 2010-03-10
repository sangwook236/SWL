#if !defined(__SWL_GRAPHICS__GRAPHICS_OBJECT__H_)
#define __SWL_GRAPHICS__GRAPHICS_OBJECT__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/IDrawable.h"
//#include "swl/graphics/ITransformable.h"
#include "swl/graphics/PickableAttrib.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

//class SWL_GRAPHICS_API GraphicsObj: public IDrawable, public ITransformable3
class SWL_GRAPHICS_API GraphicsObj: public IDrawable
{
public:
	//typedef GraphicsObj base_type;

protected:
	GraphicsObj();
	GraphicsObj(const GraphicsObj &rhs);
	virtual ~GraphicsObj() = 0;  // implemented

public:
	GraphicsObj & operator=(const GraphicsObj &rhs);

public:
	void setPickable(bool isPickable)  {  pickable_.setPickable(isPickable);  }
	bool isPickable() const  {  return pickable_.isPickable();  }

private:
	///
	PickableAttrib pickable_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GRAPHICS_OBJECT__H_
