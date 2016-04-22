#include "swl/Config.h"
#include "swl/glutil/GLCreateDisplayListVisitor.h"
#include "swl/glutil/GLShapeSceneNode.h"
#include "swl/graphics/AppearanceSceneNode.h"
#include "swl/graphics/GeometrySceneNode.h"
#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
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
// class GLCreateDisplayListVisitor

void GLCreateDisplayListVisitor::visit(const appearance_node_type &node) const
{
	throw std::runtime_error("not yet implemented");
}

void GLCreateDisplayListVisitor::visit(const geometry_node_type &node) const
{
	throw std::runtime_error("not yet implemented");
}

void GLCreateDisplayListVisitor::visit(const shape_node_type &node) const
{
	const shape_node_type::shape_type &shape = node.getShape();
	if (!shape) return;

	switch (displayListMode_)
	{
	case DLM_CREATE:
		//if (shape->isVisible() && shape->isDisplayListUsed())
		if (shape->isDisplayListUsed())
			shape->createDisplayList();
		break;
	case DLM_GENERATE_NAME:
		shape->pushDisplayList();
		break;
	case DLM_DELETE_NAME:
		shape->popDisplayList();
		break;
	}
}

}  // namespace swl
