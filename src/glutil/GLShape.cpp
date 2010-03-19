#include "swl/Config.h"
#include "swl/glutil/GLShape.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GLShape

GLShape::GLShape(const unsigned int displayListName /*= 0u*/, const bool isTransparent /*= false*/, const bool isPrintable /*= true*/, const bool isPickable /*= true*/, const attrib::PolygonMode polygonMode /*= attrib::POLYGON_FILL*/, const attrib::PolygonFace drawingFace /*= attrib::POLYGON_FACE_FRONT*/)
: base_type(isTransparent, isPrintable, isPickable, polygonMode, drawingFace),
  displayListName_(displayListName)
{
}

GLShape::GLShape(const GLShape &rhs)
: base_type(rhs),
  displayListName_(rhs.displayListName_)
{
}

GLShape::~GLShape()
{
}

GLShape & GLShape::operator=(const GLShape &rhs)
{
	if (this == &rhs) return *this;
	static_cast<base_type &>(*this) = rhs;
	// TODO [check] >>
	//displayListName_ = rhs.displayListName_;
	return *this;
}

void GLShape:processToPick() const
{
}	isDisplayListUsed() ? callDisplayList() : draw();


}  // namespace swl
