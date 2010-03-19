#include "stdafx.h"
#include "swl/Config.h"
#include "SceneGraphShape.h"
#include "swl/winview/WglFont.h"
#include "swl/glutil/GLCamera.h"
#include "swl/math/MathConstant.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <cmath>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif

#define __USE_OPENGL_DISPLAY_LIST 1


//-----------------------------------------------------------------------------
//

void Main1Shape::draw() const
{
	const GLenum drawingFace = getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT ? GL_FRONT :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_BACK ? GL_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT_AND_BACK ? GL_FRONT_AND_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_NONE ? GL_NONE : GL_FRONT)));
	const GLenum polygonMode = getPolygonMode() == swl::attrib::POLYGON_FILL ? GL_FILL :
		(getPolygonMode() == swl::attrib::POLYGON_LINE ? GL_LINE :
		(getPolygonMode() == swl::attrib::POLYGON_POINT ? GL_POINT : GL_FILL));

	// save states
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);
	glPolygonMode(drawingFace, polygonMode);
	//glPolygonMode(GL_FRONT, polygonMode);  // not working !!!

	glPushMatrix();
		//glLoadIdentity();
		glTranslatef(-250.0f, 250.0f, -250.0f);

		const double clippingPlane0[] = { 1.0, 0.0, 0.0, 100.0 };
		const double clippingPlane1[] = { -1.0, 0.0, 0.0, 300.0 };

		// draw clipping areas
		drawClippingArea(GL_CLIP_PLANE0, clippingPlane0);
		drawClippingArea(GL_CLIP_PLANE1, clippingPlane1);

		// enables clipping planes
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clippingPlane0);
		glEnable(GL_CLIP_PLANE1);
		glClipPlane(GL_CLIP_PLANE1, clippingPlane1);

		glPushName(reinterpret_cast<GLuint>(this));
			// draw a sphere
			// FIXME [uncomment] >>
			//if (PON_SPHERE == pickedObj_ || (PON_SPHERE == temporarilyPickedObj_ && isPickObjectState))
			//	glColor4f(pickedColor_[0], pickedColor_[1], pickedColor_[2], pickedColor_[3]);
			//else
				glColor4f(red(), green(), blue(), alpha());
			GL_LINE == polygonMode ? glutWireSphere(500.0, 20, 20) : glutSolidSphere(500.0, 20, 20);
		glPopName();

		// disables clipping planes
		glDisable(GL_CLIP_PLANE0);
		glDisable(GL_CLIP_PLANE1);
	glPopMatrix();

	// restore states
	//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
	glPolygonMode(drawingFace, oldPolygonMode[1]);
}

bool Main1Shape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void Main1Shape::callDisplayList() const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
}

void Main1Shape::drawClippingArea(const unsigned int clippingPlaneId, const double *clippingPlaneEqn) const
{
	glEnable(clippingPlaneId);
	glClipPlane(clippingPlaneId, clippingPlaneEqn);

	//----- rendering the mesh's clip edge
	glEnable(GL_STENCIL_TEST);
	glEnable(GL_CULL_FACE);
	glClear(GL_STENCIL_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

	// first pass: increment stencil buffer value on back faces
	glStencilFunc(GL_ALWAYS, 0, 0);
	glStencilOp(GL_KEEP, GL_KEEP, GL_INCR);
	glCullFace(GL_FRONT);  // render back faces only

	glutSolidSphere(500.0, 20, 20);

	// second pass: decrement stencil buffer value on front faces
	glStencilOp(GL_KEEP, GL_KEEP, GL_DECR);
	glCullFace(GL_BACK);  // render front faces only

	glutSolidSphere(500.0, 20, 20);

	// drawing clip planes masked by stencil buffer content
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glEnable(GL_DEPTH_TEST);
	glDisable(clippingPlaneId);
	glStencilFunc(GL_NOTEQUAL, 0, ~0); 
	// stencil test will pass only when stencil buffer value = 0. (~0 = 0x11...11)

	// rendering the plane quad. Note, it should be big enough to cover all clip edge area.
	glCullFace(GL_BACK);  // render back faces only
	glBegin(GL_QUADS);
		glColor3f(0.0f, 0.0f, 0.0f);
		// FIXME [correct] >> to be generalized
		if (clippingPlaneEqn[0] < 0.0f)
		{
			const GLfloat x = -clippingPlaneEqn[3] / clippingPlaneEqn[0];
			glVertex3f(x, -1000.0f, -1000.0f);
			glVertex3f(x, 1000.0f, -1000.0f);
			glVertex3f(x, 1000.0f, 1000.0f);
			glVertex3f(x, -1000.0f, 1000.0f);
		}
		else
		{
			const GLfloat x = -clippingPlaneEqn[3] / clippingPlaneEqn[0];
			glVertex3f(x, -1000.0f, -1000.0f);
			glVertex3f(x, -1000.0f, 1000.0f);
			glVertex3f(x, 1000.0f, 1000.0f);
			glVertex3f(x, 1000.0f, -1000.0f);
		}
	glEnd();

	glDisable(GL_STENCIL_TEST);
	glDisable(GL_CULL_FACE);
	//----- end rendering mesh's clip edge
}

//-----------------------------------------------------------------------------
//

void Main2Shape::draw() const
{
	const GLenum drawingFace = getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT ? GL_FRONT :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_BACK ? GL_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT_AND_BACK ? GL_FRONT_AND_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_NONE ? GL_NONE : GL_FRONT)));
	const GLenum polygonMode = getPolygonMode() == swl::attrib::POLYGON_FILL ? GL_FILL :
		(getPolygonMode() == swl::attrib::POLYGON_LINE ? GL_LINE :
		(getPolygonMode() == swl::attrib::POLYGON_POINT ? GL_POINT : GL_FILL));

	// save states
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);
	glPolygonMode(drawingFace, polygonMode);
	//glPolygonMode(GL_FRONT, polygonMode);  // not working !!!

	glPushMatrix();
		glTranslatef(250.0f, -250.0f, 250.0f);

		glPushName(reinterpret_cast<GLuint>(this));
			// draw a cube
			// FIXME [uncomment] >>
			//if (PON_CUBE == pickedObj_ || (PON_CUBE == temporarilyPickedObj_ && isPickObjectState))
			//	glColor4f(pickedColor_[0], pickedColor_[1], pickedColor_[2], pickedColor_[3]);
			//else
				glColor4f(red(), green(), blue(), alpha());
			GL_LINE == polygonMode ? glutWireCube(500.0) : glutSolidCube(500.0);
		glPopName();
	glPopMatrix();

	// restore states
	//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
	glPolygonMode(drawingFace, oldPolygonMode[1]);
}

bool Main2Shape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void Main2Shape::callDisplayList() const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
}

//-----------------------------------------------------------------------------
//

void GradientBackgroundShape::draw() const
{
	const GLenum drawingFace = getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT ? GL_FRONT :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_BACK ? GL_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT_AND_BACK ? GL_FRONT_AND_BACK :
		(getDrawingFace() == swl::attrib::POLYGON_FACE_NONE ? GL_NONE : GL_FRONT)));
	const GLenum polygonMode = getPolygonMode() == swl::attrib::POLYGON_FILL ? GL_FILL :
		(getPolygonMode() == swl::attrib::POLYGON_LINE ? GL_LINE :
		(getPolygonMode() == swl::attrib::POLYGON_POINT ? GL_POINT : GL_FILL));

	// save states
	const GLboolean isLighting = glIsEnabled(GL_LIGHTING);
	if (isLighting) glDisable(GL_LIGHTING);
	const GLboolean isDepthTest = glIsEnabled(GL_DEPTH_TEST);
	if (isDepthTest) glDisable(GL_DEPTH_TEST);
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);
	if (polygonMode != oldPolygonMode[1]) glPolygonMode(drawingFace, polygonMode);

	int oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);

	// save modelview matrix
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();  // reset modelview matrix

	// set to 2D orthogonal projection
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
		glLoadIdentity();  // reset projection matrix
		gluOrtho2D(0.0, 1.0, 0.0, 1.0);

		glBegin(GL_QUADS);
			glColor4f(red(), green(), blue(), alpha());
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(1.0f, 0.0f, 0.0f);

			glColor4f(topColor_.r, topColor_.g, topColor_.b, topColor_.a);
			glVertex3f(1.0f, 1.0f, 0.0f);
			glVertex3f(0.0f, 1.0f, 0.0f);
		glEnd();
	glPopMatrix();

	// restore modelview matrix
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// restore states
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

	if (polygonMode != oldPolygonMode[1])
		//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
		glPolygonMode(drawingFace, oldPolygonMode[1]);
	if (isLighting) glEnable(GL_LIGHTING);
	if (isDepthTest) glEnable(GL_DEPTH_TEST);
}

bool GradientBackgroundShape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void GradientBackgroundShape::callDisplayList() const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
}

//-----------------------------------------------------------------------------
//

void FloorShape::draw() const
{
	const float minXBound = -500.0f, maxXBound = 500.0f;
	const float minYBound = -500.0f, maxYBound = 500.0f;
	const float minZBound = -500.0f, maxZBound = 500.0f;
	const float angleThreshold = (float)std::cos(80.0 * swl::MathConstant::TO_RAD);
	const size_t lineCount = 5;
	const int lineStippleScaleFactor = 2;

	const boost::shared_ptr<view_type::camera_type> &camera = view_.topCamera();
	if (camera.get())
	{
		const GLenum drawingFace = getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT ? GL_FRONT :
			(getDrawingFace() == swl::attrib::POLYGON_FACE_BACK ? GL_BACK :
			(getDrawingFace() == swl::attrib::POLYGON_FACE_FRONT_AND_BACK ? GL_FRONT_AND_BACK :
			(getDrawingFace() == swl::attrib::POLYGON_FACE_NONE ? GL_NONE : GL_FRONT)));
		const GLenum polygonMode = getPolygonMode() == swl::attrib::POLYGON_FILL ? GL_FILL :
			(getPolygonMode() == swl::attrib::POLYGON_LINE ? GL_LINE :
			(getPolygonMode() == swl::attrib::POLYGON_POINT ? GL_POINT : GL_FILL));

		double dirX = 0.0, dirY = 0.0, dirZ = 0.0;
		camera->getEyeDirection(dirX, dirY, dirZ);

		const bool isXYPlaneShown = std::fabs((float)dirZ) >= angleThreshold;
		const bool isYZPlaneShown = std::fabs((float)dirX) >= angleThreshold;
		const bool isZXPlaneShown = std::fabs((float)dirY) >= angleThreshold;

		const bool isNegativeXYPlane = dirZ < 0.0;
		const bool isNegativeYZPlane = dirX < 0.0;
		const bool isNegativeZXPlane = dirY < 0.0;

		// save states
		const GLboolean isLighting = glIsEnabled(GL_LIGHTING);
		if (isLighting) glDisable(GL_LIGHTING);
		const GLboolean isDepthTest = glIsEnabled(GL_DEPTH_TEST);
		if (isDepthTest) glDisable(GL_DEPTH_TEST);

		const GLboolean isLineStipple = glIsEnabled(GL_LINE_STIPPLE);
		if (!isLineStipple) glEnable(GL_LINE_STIPPLE);
		GLint oldPolygonMode[2];
		glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);
		if (polygonMode != oldPolygonMode[1]) glPolygonMode(drawingFace, GL_LINE);

		//const float xmargin = 0.0f, ymargin = 0.0f, zmargin = 0.0f;
		const float marginRatio = 0.3f;
		const float xmargin = (maxXBound - minXBound) * marginRatio, ymargin = (maxYBound - minYBound) * marginRatio, zmargin = (maxZBound - minZBound) * marginRatio;
		const float margin = std::min(std::min(xmargin, ymargin), zmargin);
		const float xmin = minXBound - margin, xmax = maxXBound + margin;
		const float ymin = minYBound - margin, ymax = maxYBound + margin;
		const float zmin = minZBound - margin, zmax = maxZBound + margin;

		const float xspace = std::fabs(xmax - xmin) / float(lineCount);
		const float yspace = std::fabs(ymax - ymin) / float(lineCount);
		const float zspace = std::fabs(zmax - zmin) / float(lineCount);
		const float space = std::min(xspace, std::min(yspace, zspace));

		const float xyPlane = isNegativeXYPlane ? zmin : zmax;
		const float yzPlane = isNegativeYZPlane ? xmin : xmax;
		const float zxPlane = isNegativeZXPlane ? ymin : ymax;

		const int xstart = (int)std::ceil(xmin / space), xend = (int)std::floor(xmax / space);
		const int ystart = (int)std::ceil(ymin / space), yend = (int)std::floor(ymax / space);
		const int zstart = (int)std::ceil(zmin / space), zend = (int)std::floor(zmax / space);

		glLineStipple(lineStippleScaleFactor, 0xAAAA);
		glBegin(GL_LINES);
			// the color of a floor
			glColor4f(red(), green(), blue(), alpha());

			// xy-plane
			if (isXYPlaneShown)
			{
				//glColor4f(floorColor_[0], 0.0f, 0.0f, floorColor_[3]);

				glVertex3f(xmin, ymin, xyPlane);  glVertex3f(xmin, ymax, xyPlane);
				glVertex3f(xmax, ymin, xyPlane);  glVertex3f(xmax, ymax, xyPlane);
				glVertex3f(xmin, ymin, xyPlane);  glVertex3f(xmax, ymin, xyPlane);
				glVertex3f(xmin, ymax, xyPlane);  glVertex3f(xmax, ymax, xyPlane);
				for (int i = xstart; i <= xend; ++i)
				{
					glVertex3f(i * space, ymin, xyPlane);
					glVertex3f(i * space, ymax, xyPlane);
				}
				for (int i = ystart; i <= yend; ++i)
				{
					glVertex3f(xmin, i * space, xyPlane);
					glVertex3f(xmax, i * space, xyPlane);
				}
			}

			// yz-plane
			if (isYZPlaneShown)
			{
				//glColor4f(0.0f, floorColor_[1], 0.0f, floorColor_[3]);

				glVertex3f(yzPlane, ymin, zmin);  glVertex3f(yzPlane, ymin, zmax);
				glVertex3f(yzPlane, ymax, zmin);  glVertex3f(yzPlane, ymax, zmax);
				glVertex3f(yzPlane, ymin, zmin);  glVertex3f(yzPlane, ymax, zmin);
				glVertex3f(yzPlane, ymin, zmax);  glVertex3f(yzPlane, ymax, zmax);
				for (int i = ystart; i <= yend; ++i)
				{
					glVertex3f(yzPlane, i * space, zmin);
					glVertex3f(yzPlane, i * space, zmax);
				}
				for (int i = zstart; i <= zend; ++i)
				{
					glVertex3f(yzPlane, ymin, i * space);
					glVertex3f(yzPlane, ymax, i * space);
				}
			}

			// zx-plane
			if (isZXPlaneShown)
			{
				//glColor4f(0.0f, 0.0f, floorColor_[2], floorColor_[3]);

				glVertex3f(xmin, zxPlane, zmin);  glVertex3f(xmax, zxPlane, zmin);
				glVertex3f(xmin, zxPlane, zmax);  glVertex3f(xmax, zxPlane, zmax);
				glVertex3f(xmin, zxPlane, zmin);  glVertex3f(xmin, zxPlane, zmax);
				glVertex3f(xmax, zxPlane, zmin);  glVertex3f(xmax, zxPlane, zmax);
				for (int i = zstart; i <= zend; ++i)
				{
					glVertex3f(xmin, zxPlane, i * space);
					glVertex3f(xmax, zxPlane, i * space);
				}
				for (int i = xstart; i <= xend; ++i)
				{
					glVertex3f(i * space, zxPlane, zmin);
					glVertex3f(i * space, zxPlane, zmax);
				}
			}
		glEnd();

		// restore states
		if (polygonMode != oldPolygonMode[1])
			//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
			glPolygonMode(drawingFace, oldPolygonMode[1]);
		if (!isLineStipple) glDisable(GL_LINE_STIPPLE);

		if (isLighting) glEnable(GL_LIGHTING);
		if (isDepthTest) glEnable(GL_DEPTH_TEST);
	}
}

bool FloorShape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void FloorShape::callDisplayList() const
{
/*
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
*/
}

//-----------------------------------------------------------------------------
//

void ColorBarShape::draw() const
{
	const size_t colorDim = 3;
	const float rgb[] = {
		1.0f, 0.0f, 0.0f,
		1.0f, 0.5f, 0.0f,
		1.0f, 1.0f, 0.0f,
		0.5f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.5f,
		0.0f, 1.0f, 1.0f,
		0.0f, 0.5f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.5f, 0.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
	};
	const size_t rgbCount = sizeof(rgb) / (sizeof(rgb[0]) * colorDim);

	// save states
	const GLboolean isLighting = glIsEnabled(GL_LIGHTING);
	if (isLighting) glDisable(GL_LIGHTING);
	const GLboolean isDepthTest = glIsEnabled(GL_DEPTH_TEST);
	if (isDepthTest) glDisable(GL_DEPTH_TEST);

	int oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);

	// save modelview matrix
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();  // reset modelview matrix

	// set to 2D orthogonal projection
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
		glLoadIdentity();  // reset projection matrix
		gluOrtho2D(0.0, 1.0, 0.0, 1.0);

		const float left = 0.90f, right = 0.95f, bottom = 0.65f, top = 0.95f;
		const float dh = (top - bottom) / float(rgbCount);
		glBegin(GL_QUADS);
			float r1 = rgb[0 * colorDim];
			float g1 = rgb[0 * colorDim + 1];
			float b1 = rgb[0 * colorDim + 2];
			float h1 = bottom + 0 * dh;
			float r2, g2, b2, h2;
			for (size_t i = 1; i < rgbCount; ++i)
			{
				glColor3f(r1, g1, b1);
				glVertex3f(left, h1, 0.0f);
				glVertex3f(right, h1, 0.0f);

				r2 = rgb[i * colorDim];
				g2 = rgb[i * colorDim + 1];
				b2 = rgb[i * colorDim + 2];
				h2 = bottom + i * dh;
				glColor3f(r2, g2, b2);
				glVertex3f(right, h2, 0.0f);
				glVertex3f(left, h2, 0.0f);

				r1 = r2;
				g1 = g2;
				b1 = b2;
				h1 = h2;
			}
		glEnd();
	glPopMatrix();

	// restore modelview matrix
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// restore states
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

	if (isLighting) glEnable(GL_LIGHTING);
	if (isDepthTest) glEnable(GL_DEPTH_TEST);
}

bool ColorBarShape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void ColorBarShape::callDisplayList() const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
}

//-----------------------------------------------------------------------------
//

void CoordinateFrameShape::draw() const
{
	const boost::shared_ptr<view_type::camera_type> &camera = view_.topCamera();
	if (NULL == camera.get()) return false;
 
	const swl::Region2<int> &oldViewport = camera->getViewport();
	const swl::Region2<double> &oldViewRegion = camera->getViewRegion();

	const int dX = int(oldViewport.getWidth() * 0.10);
	const int dY = int(oldViewport.getHeight() * 0.10);
	const int size = std::max(std::max(dX, dY), 100);

	camera->setViewport(swl::Region2<int>(oldViewport.left, oldViewport.bottom, size, size));
	camera->setViewRegion(static_cast<swl::ViewCamera2 *>(camera.get())->getViewBound());
	const swl::Region2<double> &currViewRegion = camera->getCurrentViewRegion();

	// save states
	const GLboolean isLighting = glIsEnabled(GL_LIGHTING);
	if (isLighting) glDisable(GL_LIGHTING);
	const GLboolean isDepthTest = glIsEnabled(GL_DEPTH_TEST);
	if (isDepthTest) glDisable(GL_DEPTH_TEST);

	GLint oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);

	//
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		// move origin
		double eyeX(0.0), eyeY(0.0), eyeZ(0.0), dirX(0.0), dirY(0.0), dirZ(0.0);
		camera->getEyePosition(eyeX, eyeY, eyeZ);
		camera->getEyeDirection(dirX, dirY, dirZ);
		const double eyeDist = camera->getEyeDistance();
		glTranslated(eyeX + eyeDist * dirX, eyeY + eyeDist * dirY, eyeZ + eyeDist * dirZ);

		std::multimap<double, int> vals;
		vals.insert(std::make_pair(std::acos(dirX), 0));
		vals.insert(std::make_pair(std::acos(dirY), 1));
		vals.insert(std::make_pair(std::acos(dirZ), 2));
		std::multimap<double, int>::iterator it = vals.begin();
		const int order1 = it->second;  ++it;
		const int order2 = it->second;  ++it;
		const int order3 = it->second;
		const int order[] = { order1, order2, order3 };

		float length = (float)std::min(currViewRegion.getHeight(), currViewRegion.getWidth()) * 0.25f;
		if (camera->isPerspective()) length *= 2.0f / std::sqrt(3.0f);
		drawCoordinateFrame(length, order);
	glPopMatrix();
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

	// restore states
	camera->setViewRegion(oldViewRegion);
	camera->setViewport(oldViewport);

	if (isLighting) glEnable(GL_LIGHTING);
	if (isDepthTest) glEnable(GL_DEPTH_TEST);
}

bool CoordinateFrameShape::createDisplayList()
{
	glNewList(displayListName_, GL_COMPILE);
		draw();
	glEndList();
	return true;
}
void CoordinateFrameShape::callDisplayList() const
{
/*
#if defined(__USE_OPENGL_DISPLAY_LIST)
	glCallList(displayListName_);
#endif
*/
}

void CoordinateFrameShape:processToPick() const
{
}

void CoordinateFrameShape::drawCoordinateFrame(const float height, const int order[]) const
{
	const float ratio = 0.7f;  // cylinder ratio
	const float size = height * ratio;
 
	const float radius = height * 0.05f;
	const float coneRadius = radius * 2.0f;
	const float coneHeight = height * (1.0f - ratio);
	//const float letterRadius = radius * 0.5f;
	//const float letterScale = radius * 0.1f;

	GLUquadricObj *obj = gluNewQuadric();
	gluQuadricDrawStyle(obj, GLU_FILL);
	gluQuadricNormals(obj, GLU_SMOOTH);

	for (int i = 0; i < 3; ++i)
	{
		if (0 == order[i])
		{
			glPushName(reinterpret_cast<GLuint>(this));

			// x axis
			// FIXME [uncomment] >>
			//if (PON_X_AXIS == pickedObj_ || (PON_X_AXIS == temporarilyPickedObj_ && isPickObjectState))
			//	glColor4f(pickedColor_[0], pickedColor_[1], pickedColor_[2], pickedColor_[3]);
			//else
				glColor3f(1.0f, 0.0f, 0.0f);
			glPushMatrix();
				glRotated(90.0, 0.0, 1.0, 0.0);
				gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
				glTranslated(0.0, 0.0, size);
				gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
			glPopMatrix();
			// TODO [check] >>
#if defined(_UNICODE) || defined(UNICODE)
			swl::WglFont::getInstance().drawText(height, 0.0f, 0.0f, L"X");
#else
     		swl::WglFont::getInstance().drawText(height, 0.0f, 0.0f, "X");
#endif

			glPopName();
		}
		else if (1 == order[i])
		{
			glPushName(reinterpret_cast<GLuint>(this));

			// y axis
			// FIXME [uncomment] >>
			//if (PON_Y_AXIS == pickedObj_ || (PON_Y_AXIS == temporarilyPickedObj_ && isPickObjectState))
			//	glColor4f(pickedColor_[0], pickedColor_[1], pickedColor_[2], pickedColor_[3]);
			//else
				glColor3f(0.0f, 1.0f, 0.0f);
			glPushMatrix();
				glRotated(-90.0, 1.0, 0.0, 0.0);
				gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
				glTranslated(0.0, 0.0, size);
				gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
			glPopMatrix();
			// TODO [check] >>
#if defined(_UNICODE) || defined(UNICODE)
			swl::WglFont::getInstance().drawText(0.0f, height, 0.0f, L"Y");
#else
			swl::WglFont::getInstance().drawText(0.0f, height, 0.0f, "Y");
#endif

			glPopName();
		}
		else if (2 == order[i])
		{
			glPushName(reinterpret_cast<GLuint>(this));

			// z axis
			// FIXME [uncomment] >>
			//if (PON_Z_AXIS == pickedObj_ || (PON_Z_AXIS == temporarilyPickedObj_ && isPickObjectState))
			//	glColor4f(pickedColor_[0], pickedColor_[1], pickedColor_[2], pickedColor_[3]);
			//else
				glColor3f(0.0f, 0.0f, 1.0f);	
			glPushMatrix();
				gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
				glTranslated(0.0, 0.0, size);
				gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
			glPopMatrix();
			// TODO [check] >>
#if defined(_UNICODE) || defined(UNICODE)
			swl::WglFont::getInstance().drawText(0.0f, 0.0f, height, L"Z");
#else
			swl::WglFont::getInstance().drawText(0.0f, 0.0f, height, "Z");
#endif

			glPopName();
		}
	}
 
	gluDeleteQuadric(obj);
}