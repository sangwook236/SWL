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

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
AppearanceSceneNode<SceneVisitor>::AppearanceSceneNode(appearance_type &appearance, const std::wstring &name /*= std::wstring()*/)
#else
AppearanceSceneNode<SceneVisitor>::AppearanceSceneNode(appearance_type &appearance, const std::string &name /*= std::string()*/);
#endif
: base_type(name),
  appearance_(appearance)
{
}

template<typename SceneVisitor>
AppearanceSceneNode<SceneVisitor>::AppearanceSceneNode(const AppearanceSceneNode &rhs)
: base_type(rhs),
  appearance_(rhs.appearance_)
{
}

template<typename SceneVisitor>
AppearanceSceneNode<SceneVisitor>::~AppearanceSceneNode()
{
}

template<typename SceneVisitor>
AppearanceSceneNode<SceneVisitor> & AppearanceSceneNode<SceneVisitor>::operator=(const AppearanceSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	appearance_ = rhs.appearance_;
	return *this;
}

template<typename SceneVisitor>
void AppearanceSceneNode<SceneVisitor>::accept(const visitor_type &visitor) const
{
	visitor.visit(*this);
}

}  // namespace swl
