#if !defined(__SWL_WGL_VIEW_TEST__SCENE_GRAPH_SHAPE__H_)
#define __SWL_WGL_VIEW_TEST__SCENE_GRAPH_SHAPE__H_ 1


#include "swl/winview/WglViewBase.h"
#include "swl/glutil/GLShape.h"


//-----------------------------------------------------------------------------
//

class Main1Shape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	Main1Shape()
	: base_type(1u, false, true, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT_AND_BACK)
	{}

public:
	/*virtual*/ void draw() const;

private:
	void drawClippingArea(const unsigned int clippingPlaneId, const double *clippingPlaneEqn) const;
};

//-----------------------------------------------------------------------------
//

class Main2Shape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	Main2Shape()
	: base_type(1u, false, true, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT_AND_BACK)
	{}

public:
	/*virtual*/ void draw() const;
};

//-----------------------------------------------------------------------------
//

class GradientBackgroundShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	GradientBackgroundShape()
	: base_type(1u, false, false, false, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT)
	{}

public:
	/*virtual*/ void draw() const;

	void setTopColor(const float r, const float g, const float b, const float a = 1.0f)
	{  topColor_.r = r;  topColor_.g = g;  topColor_.b = b;  topColor_.a = a;  }
	void setBottomColor(const float r, const float g, const float b, const float a = 1.0f)
	{  setColor(r, g, b, a);  }

private:
	swl::Color4<float> topColor_;
};

//-----------------------------------------------------------------------------
//

class FloorShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;
	typedef swl::WglViewBase view_type;

public:
	FloorShape(view_type &view)
	: base_type(0u, false, true, false, swl::attrib::POLYGON_LINE, swl::attrib::POLYGON_FACE_FRONT),
	  view_(view)
	{}

public:
	/*virtual*/ void draw() const;

	/*virtual*/ bool createDisplayList();
	/*virtual*/ void callDisplayList() const;

private:
	view_type &view_;
};

//-----------------------------------------------------------------------------
//

class ColorBarShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	ColorBarShape()
	: base_type(1u, false, true, false, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT)
	{}

public:
	/*virtual*/ void draw() const;
};

//-----------------------------------------------------------------------------
//

class CoordinateFrameShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;
	typedef swl::WglViewBase view_type;

public:
	CoordinateFrameShape(view_type &view)
	: base_type(0u, false, false, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT),
	  view_(view)
	{}

public:
	/*virtual*/ void draw() const;

	/*virtual*/ bool createDisplayList();
	/*virtual*/ void callDisplayList() const;

	/*virtual*/ void processToPick(const int x, const int y, const int width, const int height) const;

private:
	void drawCoordinateFrame(const float height, const int order[]) const;

private:
	view_type &view_;
};


#endif  // __SWL_WGL_VIEW_TEST__SCENE_GRAPH_SHAPE__H_
