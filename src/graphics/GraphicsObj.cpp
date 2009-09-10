#include "swl/graphics/GraphicsObj.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------------
// class GraphicsObj

GraphicsObj::GraphicsObj()
: //base_type(),
  visible_(), pickable_()
{}

GraphicsObj::GraphicsObj(const GraphicsObj &rhs)
: //base_type(rhs),
  visible_(rhs.visible_), pickable_(rhs.pickable_)
{}

GraphicsObj::~GraphicsObj()
{}

GraphicsObj & GraphicsObj::operator=(const GraphicsObj &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<GraphicsObj::base_type &>(*this) = rhs;
	visible_ = rhs.visible_;
	pickable_ = rhs.pickable_;
	return *this;
}

}  // namespace swl
