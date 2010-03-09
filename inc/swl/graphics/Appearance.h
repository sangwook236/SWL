#if !defined(__SWL_GRAPHICS__APPEARANCE__H_)
#define __SWL_GRAPHICS__APPEARANCE__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Color.h"
#include "swl/graphics/VisibleAttrib.h"
#include "swl/graphics/TransparentAttrib.h"


namespace swl {

#if defined(_MSC_VER)
#pragma warning(disable:4231)
SWL_GRAPHICS_EXPORT_TEMPLATE template struct SWL_GRAPHICS_API Color4<float>;
#endif


//-----------------------------------------------------------------------------------------
// class Appearance

class SWL_GRAPHICS_API Appearance
{
public:
	//typedef Appearance base_type;
	typedef VisibleAttrib::PolygonMode PolygonMode;

public:
	Appearance();
	Appearance(const Appearance &rhs);
	virtual ~Appearance();

	Appearance & operator=(const Appearance &rhs);

public:
	/// accessor & mutator
	float & red()  {  return color_.r;  }
	float red() const  {  return color_.r;  }
	float & green()  {  return color_.g;  }
	float green() const  {  return color_.g;  }
	float & blue()  {  return color_.b;  }
	float blue() const  {  return color_.b;  }
	float & alpha()  {  return color_.a;  }
	float alpha() const  {  return color_.a;  }

	void setVisible(bool isVisible)  {  visible_.setVisible(isVisible);  }
	bool isVisible() const  {  return visible_.isVisible();  }

	void setPolygonMode(const PolygonMode polygonMode)  {  visible_.setPolygonMode(polygonMode);  }
	PolygonMode getPolygonMode() const  {  return visible_.getPolygonMode();  }

	void setTransparent(bool isTransparent)  {  transparent_.setTransparent(isTransparent);  }
	bool isTransparent() const  {  return transparent_.isTransparent();  }

private:
	Color4<float> color_;

	VisibleAttrib visible_;
	TransparentAttrib transparent_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__APPEARANCE__H_
