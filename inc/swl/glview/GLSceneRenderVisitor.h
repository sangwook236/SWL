#if !defined(__SWL_GL_VIEW__GL_SCENE_RENDER_VISITOR__H_)
#define __SWL_GL_VIEW__GL_SCENE_RENDER_VISITOR__H_ 1


#include "swl/glview/ExportGLView.h"
#include "swl/graphics/ISceneVisitor.h"


namespace swl {

//--------------------------------------------------------------------------
// class GLSceneRenderVisitor

class SWL_GL_VIEW_API GLSceneRenderVisitor: public ISceneVisitor
{
public:
	//typedef ISceneVisitor base_type;

public:
	/*virtual*/ void visit(const AppearanceSceneNode &node) const;
	/*virtual*/ void visit(const GeometrySceneNode &node) const;
	/*virtual*/ void visit(const TransformSceneNode &node) const;

};

}  // namespace swl


#endif  // __SWL_GL_VIEW__GL_SCENE_RENDER_VISITOR__H_
