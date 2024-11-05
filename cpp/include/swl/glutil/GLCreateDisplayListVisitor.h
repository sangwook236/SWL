#if !defined(__SWL_GL_UTIL__GL_CREATE_DISPLAY_LIST_VISITOR__H_)
#define __SWL_GL_UTIL__GL_CREATE_DISPLAY_LIST_VISITOR__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include "swl/glutil/IGLSceneVisitor.h"


namespace swl {

//--------------------------------------------------------------------------
// class GLCreateDisplayListVisitor

class SWL_GL_UTIL_API GLCreateDisplayListVisitor: public IGLSceneVisitor
{
public:
	//typedef IGLSceneVisitor base_type;

public:
	enum DisplayListMode { DLM_CREATE, DLM_GENERATE_NAME, DLM_DELETE_NAME };

public:
	GLCreateDisplayListVisitor(const DisplayListMode displayListMode)
	: displayListMode_(displayListMode)
	{}

public:
	/*virtual*/ void visit(const appearance_node_type &node) const;
	/*virtual*/ void visit(const geometry_node_type &node) const;
	/*virtual*/ void visit(const shape_node_type &node) const;

	/*virtual*/ void visit(const transform_node_type & /*node*/) const  {}

private:
	const DisplayListMode displayListMode_;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_CREATE_DISPLAY_LIST_VISITOR__H_
