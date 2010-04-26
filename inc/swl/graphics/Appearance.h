#if !defined(__SWL_GRAPHICS__APPEARANCE__H_)
#define __SWL_GRAPHICS__APPEARANCE__H_ 1


#include "swl/graphics/ExportGraphics.h"
#include "swl/graphics/Color.h"


namespace swl {

//#if defined(_MSC_VER)
//#pragma warning(disable:4231)
//SWL_GRAPHICS_EXPORT_TEMPLATE template struct SWL_GRAPHICS_API Color4<float>;
//#endif


//-----------------------------------------------------------------------------------------
//

namespace attrib {
	enum PolygonFace { POLYGON_FACE_NONE, POLYGON_FACE_FRONT, POLYGON_FACE_BACK, POLYGON_FACE_FRONT_AND_BACK };
	enum PolygonMode { POLYGON_POINT, POLYGON_LINE, POLYGON_FILL };
	//enum ShadingMode { FLAT_SHADING, SMOOTH_SHADING };
}  // namespace attrib

//-----------------------------------------------------------------------------------------
// class Appearance

class SWL_GRAPHICS_API Appearance
{
public:
	//typedef Appearance base_type;

public:
	Appearance(const bool isVisible = true, const bool isTransparent = false, const attrib::PolygonMode polygonMode = attrib::POLYGON_FILL, const attrib::PolygonFace drawingFace = attrib::POLYGON_FACE_FRONT);
	Appearance(const Appearance &rhs);
	virtual ~Appearance();

	Appearance & operator=(const Appearance &rhs);

public:
	void setColor(const float r, const float g, const float b, const float a = 1.0f)
	{  color_.r = r;  color_.g = g;  color_.b = b;  color_.a = a;  }
	void getColor(float &r, float &g, float &b)
	{  r = color_.r;  g = color_.g;  b = color_.b;  }
	void getColor(float &r, float &g, float &b, float &a)
	{  r = color_.r;  g = color_.g;  b = color_.b;  a = color_.a;  }
	float & red()  {  return color_.r;  }
	float red() const  {  return color_.r;  }
	float & green()  {  return color_.g;  }
	float green() const  {  return color_.g;  }
	float & blue()  {  return color_.b;  }
	float blue() const  {  return color_.b;  }
	float & alpha()  {  return color_.a;  }
	float alpha() const  {  return color_.a;  }

	void setVisible(const bool isVisible)  {  isVisible_ = isVisible;  }
	bool isVisible() const  {  return isVisible_;  }

	void setTransparent(const bool isTransparent)  {  isTransparent_ = isTransparent;  }
	bool isTransparent() const  {  return isTransparent_;  }

	void setPolygonMode(const attrib::PolygonMode polygonMode)  {  polygonMode_ = polygonMode;  }
	attrib::PolygonMode getPolygonMode() const  {  return polygonMode_;  }

	void setDrawingFace(const attrib::PolygonFace drawingFace)  {  drawingFace_ = drawingFace;  }
	attrib::PolygonFace getDrawingFace() const  {  return drawingFace_;  }

private:
	Color4<float> color_;

	bool isVisible_;
	bool isTransparent_;

	attrib::PolygonMode polygonMode_;
	attrib::PolygonFace drawingFace_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__APPEARANCE__H_
