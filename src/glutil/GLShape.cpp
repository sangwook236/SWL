#include "swl/Config.h"
#include "swl/glutil/GLShape.h"
#if defined(WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GLShape

GLShape::GLShape(const unsigned int displayListCount, const bool isTransparent /*= false*/, const bool isPrintable /*= true*/, const bool isPickable /*= true*/, const attrib::PolygonMode polygonMode /*= attrib::POLYGON_FILL*/, const attrib::PolygonFace drawingFace /*= attrib::POLYGON_FACE_FRONT*/)
: base_type(isTransparent, isPrintable, isPickable, polygonMode, drawingFace),
  GLDisplayListCallableInterface(displayListCount)
{
}

GLShape::GLShape(const GLShape &rhs)
: base_type(rhs),
  GLDisplayListCallableInterface(rhs)
{
}

GLShape::~GLShape()
{
}

GLShape & GLShape::operator=(const GLShape &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	//static_cast<GLDisplayListCallableInterface &>(*this) = rhs;
	return *this;
}

bool GLShape::createDisplayList()
{
	glNewList(getDisplayListNameBase(), GL_COMPILE);
		draw();
	glEndList();
	return true;
}

void GLShape::callDisplayList() const
{
	glCallList(getDisplayListNameBase());
}

void GLShape::processToPick(const int /*x*/, const int /*y*/, const int /*width*/, const int /*height*/) const
{
	isDisplayListUsed() ? callDisplayList() : draw();
}

}  // namespace swl
