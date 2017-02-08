#include "swl/Config.h"
#include "swl/glutil/GLPickObjectVisitor.h"
#include "swl/glutil/GLShapeSceneNode.h"
#include "swl/graphics/AppearanceSceneNode.h"
#include "swl/graphics/GeometrySceneNode.h"
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>
#include <stdexcept>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GLPickObjectVisitor

void GLPickObjectVisitor::visit(const appearance_node_type &node) const
{
	throw std::runtime_error("Not yet implemented");
}

void GLPickObjectVisitor::visit(const geometry_node_type &node) const
{
	throw std::runtime_error("Not yet implemented");
}

void GLPickObjectVisitor::visit(const shape_node_type &node) const
{
	const shape_node_type::shape_type &shape = node.getShape();
	if (!shape) return;

	if (!shape->isVisible() || !shape->isPickable()) return;

	shape->processToPick(x_, y_, width_, height_);
}

}  // namespace swl
