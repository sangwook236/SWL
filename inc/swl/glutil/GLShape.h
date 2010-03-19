#if !defined(__SWL_GL_UTIL__GL_SHAPE__H_)
#define __SWL_GL_UTIL__GL_SHAPE__H_ 1


#include "swl/glutil/ExportGLUtil.h"
#include "swl/glutil/IGLDisplayListCallable.h"
#include "swl/graphics/Shape.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class GLShape

class SWL_GL_UTIL_API GLShape: public Shape, public IGLDisplayListCallable
{
public:
	typedef Shape base_type;

public:
	GLShape(const unsigned int displayListName = 0u, const bool isTransparent = false, const bool isPrintable = true, const bool isPickable = true, const attrib::PolygonMode polygonMode = attrib::POLYGON_FILL, const attrib::PolygonFace drawingFace = attrib::POLYGON_FACE_FRONT);
	GLShape(const GLShape &rhs);
	virtual ~GLShape();

	GLShape & operator=(const GLShape &rhs);

public:
	//
	bool isDisplayListUsed() const  {  return 0 != displayListName_;  }

protected:
	//
	const unsigned int displayListName_;
};

}  // namespace swl


#endif  // __SWL_GL_UTIL__GL_SHAPE__H_
