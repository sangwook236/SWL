#if !defined(__SWL_GRAPHICS__SHAPE__H_)
#define __SWL_GRAPHICS__SHAPE__H_ 1


#include "swl/graphics/GraphicsObj.h"
#include "swl/graphics/GeometryPool.h"
#include "swl/graphics/Appearance.h"


namespace swl {

//-----------------------------------------------------------------------------------------
// class Shape

class SWL_GRAPHICS_API Shape: public GraphicsObj
{
public:
	typedef GraphicsObj						base_type;
	typedef GeometryPool::geometry_id_type	geometry_id_type;
	typedef GeometryPool::geometry_type		geometry_type;
	typedef Appearance						appearance_type;

public:
	Shape(const bool isTransparent = false, const bool isPrintable = true, const bool isPickable = true, const attrib::PolygonMode polygonMode = attrib::POLYGON_FILL, const attrib::PolygonFace drawingFace = attrib::POLYGON_FACE_FRONT);
	Shape(const Shape &rhs);
	virtual ~Shape();

	Shape & operator=(const Shape &rhs);

public:
	//
	geometry_id_type & getGeometryId()  {  return geometryId_;  }
	const geometry_id_type & getGeometryId() const  {  return geometryId_;  }

	geometry_type getGeometry() const;

	//
	void setColor(const float r, const float g, const float b, const float a = 1.0f)
	{  appearance_.setColor(r, g, b, a);;  }
	void getColor(float &r, float &g, float &b)
	{  appearance_.getColor(r, g, b);  }
	void getColor(float &r, float &g, float &b, float &a)
	{  appearance_.getColor(r, g, b, a);  }
	float & red()  {  return appearance_.red();  }
	float red() const  {  return appearance_.red();  }
	float & green()  {  return appearance_.green();  }
	float green() const  {  return appearance_.green();  }
	float & blue()  {  return appearance_.blue();  }
	float blue() const  {  return appearance_.blue();  }
	float & alpha()  {  return appearance_.alpha();  }
	float alpha() const  {  return appearance_.alpha();  }

	void setVisible(const bool isVisible)  {  appearance_.setVisible(isVisible);  }
	bool isVisible() const  {  return appearance_.isVisible();  }

	void setTransparent(const bool isTransparent)  {  appearance_.setTransparent(isTransparent);  }
	bool isTransparent() const  {  return appearance_.isTransparent();  }

	void setPolygonMode(const attrib::PolygonMode polygonMode)  {  appearance_.setPolygonMode(polygonMode);  }
	attrib::PolygonMode getPolygonMode() const  {  return appearance_.getPolygonMode();  }

	void setDrawingFace(const attrib::PolygonFace drawingFace)  {  appearance_.setDrawingFace(drawingFace);  }
	attrib::PolygonFace getDrawingFace() const  {  return appearance_.getDrawingFace();  }

private:
	//
	geometry_id_type geometryId_;
	appearance_type appearance_;
};

}  // namespace swl


#endif  // __SWL_GRAPHICS__SHAPE__H_
