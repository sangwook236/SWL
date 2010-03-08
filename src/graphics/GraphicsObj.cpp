#include "swl/Config.h"
#include "swl/graphics/GraphicsObj.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

GraphicsObj::GraphicsObj()
: //base_type(),
  pickable_()
{}

GraphicsObj::GraphicsObj(const GraphicsObj &rhs)
: //base_type(rhs),
  pickable_(rhs.pickable_)
{}

GraphicsObj::~GraphicsObj()
{}

GraphicsObj & GraphicsObj::operator=(const GraphicsObj &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	pickable_ = rhs.pickable_;
	return *this;
}

}  // namespace swl
