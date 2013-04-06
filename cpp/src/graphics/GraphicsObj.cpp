#include "swl/Config.h"
#include "swl/graphics/GraphicsObj.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

GraphicsObj::GraphicsObj(const bool isPrintable, const bool isPickable)
: //base_type(),
  PickableInterface(isPickable),
  isPrintable_(isPrintable)
{}

GraphicsObj::GraphicsObj(const GraphicsObj &rhs)
: //base_type(rhs),
  PickableInterface(rhs),
  isPrintable_(rhs.isPrintable_)
{}

GraphicsObj::~GraphicsObj()
{}

GraphicsObj & GraphicsObj::operator=(const GraphicsObj &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	static_cast<PickableInterface &>(*this) = rhs;
	isPrintable_ = rhs.isPrintable_;
	return *this;
}

}  // namespace swl
