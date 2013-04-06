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
	GLPickObjectVisitor(const int x, const int y, const int width, const int height)
	: x_(x), y_(y), width_(width), height_(height)
	{}

public:
	/*virtual*/ void visit(const appearance_node_type &node) const;
	/*virtual*/ void visit(const geometry_node_type &node) const;
	/*virtual*/ void visit(const shape_node_type &node) const;

	/*virtual*/ void visit(const transform_node_type & /*node*/) const  {}

private:
	const int x_, y_;
	const int width_, height_;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_PICK_OBJECT_VISITOR__H_
