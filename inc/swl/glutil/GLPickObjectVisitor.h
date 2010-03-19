#if !defined(__SWL_GL_UTIL__GL_PICK_OBJECT_VISITOR__H_)
#define __SWL_GL_UTIL__GL_PICK_OBJECT_VISITOR__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include "swl/glutil/IGLSceneVisitor.h"
#include <boost/smart_ptr.hpp>


namespace swl {

//--------------------------------------------------------------------------
// class GLPickObjectVisitor

class SWL_GL_UTIL_API GLPickObjectVisitor: public IGLSceneVisitor
{
public:
	//typedef IGLSceneVisitor base_type;

public:
	GLPickObjectVisitor()
	{}

public:
	/*virtual*/ void visit(const appearance_node_type &node) const;
	/*virtual*/ void visit(const geometry_node_type &node) const;
	/*virtual*/ void visit(const shape_node_type &node) const;

	/*virtual*/ void visit(const transform_node_type & /*node*/) const  {}
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_PICK_OBJECT_VISITOR__H_
