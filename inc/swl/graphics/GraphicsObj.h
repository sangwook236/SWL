#if !defined(__SWL_GRAPHICS__GRAPHICS_OBJECT__H_)
#define __SWL_GRAPHICS__GRAPHICS_OBJECT__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/IDrawable.h"
#include "swl/graphics/ITransformable.h"
#include "swl/graphics/VisibleAttrib.h"
#include "swl/graphics/PickableAttrib.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

class SWL_GRAPHICS_API GraphicsObj: public IDrawable, public ITransformable
{
public:
	//typedef GraphicsObj base_type;
	typedef VisibleAttrib::PolygonMode PolygonMode;

protected:
	GraphicsObj();
	GraphicsObj(const GraphicsObj& rhs);
	virtual ~GraphicsObj() = 0;  // implemented

public:
	GraphicsObj& operator=(const GraphicsObj& rhs);

public:
	void setVisible(bool isVisible)  {  visible_.setVisible(isVisible);  }
	bool isVisible() const  {  return visible_.isVisible();  }

	void setPolygonMode(const PolygonMode polygonMode)  {  visible_.setPolygonMode(polygonMode);  }
	PolygonMode getPolygonMode() const  {  return visible_.getPolygonMode();  }

	void setPickable(bool isPickable)  {  pickable_.setPickable(isPickable);  }
	bool isPickable() const  {  return pickable_.isPickable();  }

private:
	///
	VisibleAttrib visible_;
	PickableAttrib pickable_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GRAPHICS_OBJECT__H_
