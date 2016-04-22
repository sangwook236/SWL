#include "swl/Config.h"
#include "swl/glutil/GLPrintSceneVisitor.h"
#include "swl/glutil/GLShapeSceneNode.h"
#include "swl/graphics/AppearanceSceneNode.h"
#include "swl/graphics/GeometrySceneNode.h"
#include "swl/graphics/TransformSceneNode.h"
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
// class GLPrintSceneVisitor

void GLPrintSceneVisitor::visit(const appearance_node_type &node) const
{
	std::runtime_error("not yet implemented");
}

void GLPrintSceneVisitor::visit(const geometry_node_type &node) const
{
	std::runtime_error("not yet implemented");
}

void GLPrintSceneVisitor::visit(const shape_node_type &node) const
{
	const shape_node_type::shape_type &shape = node.getShape();
	if (!shape) return;

	if (!shape->isVisible() || !shape->isPrintable()) return;
	if ((RENDER_OPAQUE_OBJECTS == renderMode_ && shape->isTransparent()) ||
		(RENDER_TRANSPARENT_OBJECTS == renderMode_ && !shape->isTransparent()))
		return;

	// FIXME [modify] >>
#if 0
	const shape_node_type::geometry_type &geometry = shape->getGeometry();
	const shape_node_type::appearance_type::PolygonMode &polygonMode = shape->getPolygonMode();

	glBegin();
		glColor4f(shape->red(), shape->green(), shape->blue(), shape->alpha());
		if (geometry) geometry->draw();
	glEnd();
#else
	//shape->isDisplayListUsed() ? shape->callDisplayList() : shape->draw();
	isPickingState_ || !shape->isDisplayListUsed() ? shape->draw() : shape->callDisplayList();
#endif
}

void GLPrintSceneVisitor::visit(const transform_node_type &node) const
{
	const transform_node_type::transform_type &tranform = node.getTransform();
	double tmat[16] = { 0.0f, };
	tranform.get(tmat);

	glPushMatrix();
		glMultMatrixd(tmat);
		node.traverse(*this);
	glPopMatrix();
}

}  // namespace swl
