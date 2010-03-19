#include "swl/Config.h"
#include "swl/glutil/GLShapeSceneNode.h"
#include "swl/graphics/ISceneVisitor.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GLShapeSceneNode

template<typename SceneVisitor>
#if defined(UNICODE) || defined(_UNICODE)
GLShapeSceneNode<SceneVisitor>::GLShapeSceneNode(shape_type &shape, const std::wstring &name /*= std::wstring()*/)
#else
GLShapeSceneNode<SceneVisitor>::GLShapeSceneNode(shape_type &shape, const std::string &name /*= std::string()*/);
#endif
: base_type(name),
  shape_(shape)
{
}

template<typename SceneVisitor>
GLShapeSceneNode<SceneVisitor>::GLShapeSceneNode(const GLShapeSceneNode &rhs)
: base_type(rhs),
  shape_(rhs.shape_)
{
}

template<typename SceneVisitor>
GLShapeSceneNode<SceneVisitor>::~GLShapeSceneNode()
{
}

template<typename SceneVisitor>
GLShapeSceneNode<SceneVisitor> & GLShapeSceneNode<SceneVisitor>::operator=(const GLShapeSceneNode &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	shape_ = rhs.shape_;
	return *this;
}

void GLShapeSceneNode<SceneVisitor>::accept(const visitor_type &visitor) const
{
	visitor.visit(*this);
}

}  // namespace swl
