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
	enum RenderMode { RENDER_OPAQUE_OBJECTS, RENDER_TRANSPARENT_OBJECTS, SELECT_OBJECTS };

public:
	GLSceneRenderVisitor(const RenderMode renderMode)
	: renderMode_(renderMode)
	{}

public:
	/*virtual*/ void visit(const AppearanceSceneNode &node) const;
	/*virtual*/ void visit(const GeometrySceneNode &node) const;
	/*virtual*/ void visit(const ShapeSceneNode &node) const;

	/*virtual*/ void visit(const TransformSceneNode &node) const;

private:
	const RenderMode renderMode_;
};

}  // namespace swl


#endif  // __SWL_GL_VIEW__GL_SCENE_RENDER_VISITOR__H_
