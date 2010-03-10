#include "swl/Config.h"
#include "swl/graphics/AppearanceSceneNode.h"
#include "swl/graphics/ISceneVisitor.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class AppearanceSceneNode

AppearanceSceneNode::AppearanceSceneNode(AppearanceSceneNode::appearance_type &appearance)
: base_type(),
  appearance_(appearance)
{
}

AppearanceSceneNode::AppearanceSceneNode(const AppearanceSceneNode &rhs)
: base_type(rhs),
  appearance_(rhs.appearance_)
{
}

AppearanceSceneNode::~AppearanceSceneNode()
{
}

AppearanceSceneNode & AppearanceSceneNode::operator=(const AppearanceSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	appearance_ = rhs.appearance_;
	return *this;
}

void AppearanceSceneNode::accept(const ISceneVisitor &visitor) const
{
	visitor.visit(*this);
}

}  // namespace swl
