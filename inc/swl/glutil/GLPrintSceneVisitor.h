#if !defined(__SWL_GL_UTIL__GL_PRINT_SCENE_VISITOR__H_)
#define __SWL_GL_UTIL__GL_PRINT_SCENE_VISITOR__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include "swl/glutil/IGLSceneVisitor.h"


namespace swl {

//--------------------------------------------------------------------------
// class GLPrintSceneVisitor

class SWL_GL_UTIL_API GLPrintSceneVisitor: public IGLSceneVisitor
{
public:
	//typedef IGLSceneVisitor base_type;

public:
	enum RenderMode { RENDER_OPAQUE_OBJECTS, RENDER_TRANSPARENT_OBJECTS, SELECT_OBJECTS };

public:
	GLPrintSceneVisitor(const RenderMode renderMode)
	: renderMode_(renderMode)
	{}

public:
	/*virtual*/ void visit(const appearance_node_type &node) const;
	/*virtual*/ void visit(const geometry_node_type &node) const;
	/*virtual*/ void visit(const shape_node_type &node) const;

	/*virtual*/ void visit(const transform_node_type &node) const;

private:
	const RenderMode renderMode_;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_PRINT_SCENE_VISITOR__H_
