#include "swl/Config.h"
#include "swl/graphics/Appearance.h"


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class Appearance

Appearance::Appearance(const bool isVisible /*= true*/, const bool isTransparent /*= false*/, const attrib::PolygonMode polygonMode /*= attrib::POLYGON_FILL*/, const attrib::PolygonFace drawingFace /*= attrib::POLYGON_FACE_FRONT*/)
: //base_type(),
  color_(), isVisible_(isVisible), isTransparent_(isTransparent), polygonMode_(polygonMode), drawingFace_(drawingFace)
{
}

Appearance::Appearance(const Appearance &rhs)
: //base_type(rhs),
  color_(rhs.color_), isVisible_(rhs.isVisible_), isTransparent_(rhs.isTransparent_), polygonMode_(rhs.polygonMode_), drawingFace_(rhs.drawingFace_)
{
}

Appearance::~Appearance()
{
}

Appearance & Appearance::operator=(const Appearance &rhs)
{
	if (this == &rhs) return *this;
	//static_cast<base_type &>(*this) = rhs;
	color_ = rhs.color_;
	isVisible_ = rhs.isVisible_;
	isTransparent_ = rhs.isTransparent_;
	polygonMode_ = rhs.polygonMode_;
	drawingFace_ = rhs.drawingFace_;
	return *this;
}

}  // namespace swl
