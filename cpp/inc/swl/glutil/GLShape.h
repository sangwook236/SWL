#if !defined(__SWL_GL_UTIL__GL_SHAPE__H_)
#define __SWL_GL_UTIL__GL_SHAPE__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include "swl/glutil/GLDisplayListCallableInterface.h"
#include "swl/graphics/Shape.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class GLShape

class SWL_GL_UTIL_API GLShape: public Shape, public GLDisplayListCallableInterface
{
public:
	typedef Shape base_type;

public:
	GLShape(const unsigned int displayListCount, const bool isTransparent = false, const bool isPrintable = true, const bool isPickable = true, const attrib::PolygonMode polygonMode = attrib::POLYGON_FILL, const attrib::PolygonFace drawingFace = attrib::POLYGON_FACE_FRONT);
	GLShape(const GLShape &rhs);
	virtual ~GLShape();

	GLShape & operator=(const GLShape &rhs);

public:
	/*virtual*/ bool createDisplayList();
	/*virtual*/ void callDisplayList() const;

	/*virtual*/ void processToPick(const int x, const int y, const int width, const int height) const;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_SHAPE__H_
