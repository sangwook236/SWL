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
  isPrintable_(isPrintable), isPickable_(isPickable)
{}

GraphicsObj::GraphicsObj(const GraphicsObj &rhs)
: //base_type(rhs),
  isPrintable_(rhs.isPrintable_), isPickable_(rhs.isPickable_)
{}

GraphicsObj::~GraphicsObj()
{}

GraphicsObj & GraphicsObj::operator=(const GraphicsObj &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	isPrintable_ = rhs.isPrintable_;
	isPickable_ = rhs.isPickable_;
	return *this;
}

}  // namespace swl
