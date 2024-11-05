#if !defined(__SWL_GRAPHICS__PICKABLE_INTERFACE__H_)
#define __SWL_GRAPHICS__PICKABLE_INTERFACE__H_ 1


#include "swl/graphics/ExportGraphics.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// struct PickableInterface: mix-in style class

struct SWL_GRAPHICS_API PickableInterface
{
public:
	//typedef PickableInterface base_type;

protected:
	PickableInterface(const bool isPickable);
	PickableInterface(const PickableInterface &rhs);
	virtual ~PickableInterface();

public:
	PickableInterface & operator=(const PickableInterface &rhs);

public:
	//
	virtual void processToPick(const int x, const int y, const int width, const int height) const = 0;

	//
	void setPickable(const bool isPickable)  {  isPickable_ = isPickable;  }
	bool isPickable() const  {  return isPickable_;  }

private:
	bool isPickable_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__PICKABLE_INTERFACE__H_
