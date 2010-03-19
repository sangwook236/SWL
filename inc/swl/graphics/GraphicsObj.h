#if !defined(__SWL_GRAPHICS__GRAPHICS_OBJECT__H_)
#define __SWL_GRAPHICS__GRAPHICS_OBJECT__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/IDrawable.h"
#include "swl/graphics/IPickable.h"
//#include "swl/graphics/ITransformable.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

//class SWL_GRAPHICS_API GraphicsObj: public IDrawable, public IPickable, public ITransformable3
class SWL_GRAPHICS_API GraphicsObj: public IDrawable, public IPickable
{
public:
	//typedef GraphicsObj base_type;

protected:
	GraphicsObj(const bool isPrintable, const bool isPickable);
	GraphicsObj(const GraphicsObj &rhs);
	virtual ~GraphicsObj() = 0;  // implemented

public:
	GraphicsObj & operator=(const GraphicsObj &rhs);

public:
	void setPrintable(const bool isPrintable)  {  isPrintable_ = isPrintable;  }
	bool isPrintable() const {  return isPrintable_;  }

	void setPickable(const bool isPickable)  {  isPickable_ = isPickable;  }
	bool isPickable() const {  return isPickable_;  }

private:
	//
	bool isPrintable_;
	bool isPickable_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__GRAPHICS_OBJECT__H_
