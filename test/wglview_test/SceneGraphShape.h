#if !defined(__SWL_WGL_VIEW_TEST__SCENE_GRAPH_SHAPE__H_)
#define __SWL_WGL_VIEW_TEST__SCENE_GRAPH_SHAPE__H_ 1


#include "swl/winview/WglViewBase.h"
#include "swl/glutil/GLShape.h"
#include <boost/smart_ptr.hpp>


//-----------------------------------------------------------------------------
//

class ClippedSphereShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	ClippedSphereShape()
	: base_type(1u, false, true, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT_AND_BACK)
	{}

public:
	/*virtual*/ void draw() const;

private:
	void drawClippingArea(const unsigned int clippingPlaneId, const double *clippingPlaneEqn) const;
};

//-----------------------------------------------------------------------------
//

class SimpleCubeShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	SimpleCubeShape()
	: base_type(1u, false, true, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT_AND_BACK)
	{}

public:
	/*virtual*/ void draw() const;
};

//-----------------------------------------------------------------------------
//

class ColoredMeshShape: public swl::GLShape
{
public:
	typedef swl::GLShape base_type;

public:
	ColoredMeshShape()
	: base_type(1u, false, true, true, swl::attrib::POLYGON_FILL, swl::attrib::POLYGON_FACE_FRONT_AND_BACK),
	  meshWidth_(320), meshHeight_(240), paletteSize_(256), paletteColorDim_(3),
	  meshMinValue_(0.0f), meshMaxValue_(0.0f),
	  xOffset_(0.0f), yOffset_(0.0f), zOffset_(0.0f), zScaleFactor_(1.0f)
	{
		loadMesh();
	}

public:
	/*virtual*/ void draw() const;

protected:
	void loadMesh();
	void calculateNormal(const float vx1, const float vy1, const float vz1, const float vx2, const float vy2, const float vz2, float &nx, float &ny, float &nz) const;

protected:
	const size_t meshWidth_, meshHeight_;
	const size_t paletteSize_;
	const size_t paletteColorDim_;

	boost::scoped_array<float> mesh_;
	boost::scoped_array<unsigned char> meshColorIndexes_;
	boost::scoped_array<unsigned char> palette_;

	float meshMinValue_, meshMaxValue_;

	const float xOffset_;
	const float yOffset_;
	const float zOffset_;
	const float zScaleFactor_;
};

//-----------------------------------------------------------------------------
//

class TexturedMeshShape: public ColoredMeshShape
{
public:
	typedef ColoredMeshShape base_type;

public:
	TexturedMeshShape();
	~TexturedMeshShape();

public:
	/*virtual*/ void draw() const;
	/*virtual*/ bool createDisplayList();

private:
	void createTexture();

	void drawTexturedMesh() const;
	void drawTexture() const;
	void drawMesh() const;

protected:
	const size_t texWidth_, texHeight_;

	static const size_t texCount_ = 1;
	unsigned int textureObjs_[texCount_];
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
