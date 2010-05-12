#include "stdafx.h"
#include "swl/Config.h"
#include "SceneGraphShape.h"
#include "swl/winview/WglFont.h"
#include "swl/glutil/GLCamera.h"
#include "swl/graphics/ObjectPickerMgr.h"
#include "swl/math/MathConstant.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <fstream>
#include <iostream>
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


//-----------------------------------------------------------------------------
//

void TrimmedSphereShape::draw() const
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
		drawClippingRegion(GL_CLIP_PLANE0, clippingPlane0);
		drawClippingRegion(GL_CLIP_PLANE1, clippingPlane1);

		// enables clipping planes
		glEnable(GL_CLIP_PLANE0);
		glClipPlane(GL_CLIP_PLANE0, clippingPlane0);
		glEnable(GL_CLIP_PLANE1);
		glClipPlane(GL_CLIP_PLANE1, clippingPlane1);

		const GLuint id = reinterpret_cast<GLuint>(this);
		glPushName(id);
			// draw a sphere
			if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(id))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else if (swl::ObjectPickerMgr::getInstance().isPickedObject(id))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else glColor4f(red(), green(), blue(), alpha());
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

void TrimmedSphereShape::drawClippingRegion(const unsigned int clippingPlaneId, const double *clippingPlaneEqn) const
{
	glEnable(clippingPlaneId);
	glClipPlane(clippingPlaneId, clippingPlaneEqn);

	//----- rendering the mesh_'s clip edge
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
	//----- end rendering mesh_'s clip edge
}

//-----------------------------------------------------------------------------
//

void SimpleCubeShape::draw() const
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

		const GLuint id = reinterpret_cast<GLuint>(this);
		glPushName(id);
			// draw a cube
			if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(id))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else if (swl::ObjectPickerMgr::getInstance().isPickedObject(id))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else glColor4f(red(), green(), blue(), alpha());
			GL_LINE == polygonMode ? glutWireCube(500.0) : glutSolidCube(500.0);
		glPopName();
	glPopMatrix();

	// restore states
	//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
	glPolygonMode(drawingFace, oldPolygonMode[1]);
}

//-----------------------------------------------------------------------------
//

void ColoredMeshShape::draw() const
{
	if (!mesh_ || !meshColorIndexes_ || !palette_) return;

	//float nx, ny, nz;
	float r, g, b;
	size_t colorIndex;

	// save states
	glPushAttrib(GL_LIGHTING_BIT);

	// set attributes
	glDisable(GL_LIGHTING);

	const GLuint id = reinterpret_cast<GLuint>(this);
	glPushName(id);
		bool useObjectColor = false;
		swl::ObjectPickerMgr::color_type objectColor;
		if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(id))
		{
			objectColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
			useObjectColor = true;
		}
		else if (swl::ObjectPickerMgr::getInstance().isPickedObject(id))
		{
			objectColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
			useObjectColor = true;
		}

		float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, temp1, temp2, temp3, temp4;
		glBegin(GL_QUADS);
			for (size_t h = 0; h < meshHeight_ - 1; ++h)
				for (size_t w = 0; w < meshWidth_ - 1; ++w)
				{
					x1 = float(w) + xOffset_;
					x2 = float(w+1) + xOffset_;
					x3 = float(w+1) + xOffset_;
					x4 = float(w) + xOffset_;
#if 0
					y1 = float(h) + yOffset_;
					y2 = float(h) + yOffset_;
					y3 = float(h+1) + yOffset_;
					y4 = float(h+1) + yOffset_;
#else
					// flipped image
					y1 = float(meshHeight_ - h) + yOffset_;
					y2 = float(meshHeight_ - h) + yOffset_;
					y3 = float(meshHeight_ - (h+1)) + yOffset_;
					y4 = float(meshHeight_ - (h+1)) + yOffset_;
#endif

					temp1 = mesh_[h * meshWidth_ + w];
					z1 = temp1 * zScaleFactor_ + zOffset_;
					temp2 = mesh_[h * meshWidth_ + (w+1)];
					z2 = temp2 * zScaleFactor_ + zOffset_;
					temp3 = mesh_[(h+1) * meshWidth_ + (w+1)];
					z3 = temp3 * zScaleFactor_ + zOffset_;
					temp4 = mesh_[(h+1) * meshWidth_ + w];
					z4 = temp4 * zScaleFactor_ + zOffset_;

					//glEdgeFlag(GL_TRUE);

					//calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					if (useObjectColor) glColor4f(objectColor.r, objectColor.g, objectColor.b, objectColor.a);
					else
					{
						colorIndex = meshColorIndexes_[h * meshWidth_ + w] * paletteColorDim_;
						r = palette_[colorIndex] / 255.0f;
						g = palette_[colorIndex + 1] / 255.0f;
						b = palette_[colorIndex + 2] / 255.0f;
						glColor3f(r, g, b);
					}
					//glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					if (useObjectColor) glColor4f(objectColor.r, objectColor.g, objectColor.b, objectColor.a);
					else
					{
						colorIndex = meshColorIndexes_[h * meshWidth_ + (w+1)] * paletteColorDim_;
						r = palette_[colorIndex] / 255.0f;
						g = palette_[colorIndex + 1] / 255.0f;
						b = palette_[colorIndex + 2] / 255.0f;
						glColor3f(r, g, b);
					}
					//glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					if (useObjectColor) glColor4f(objectColor.r, objectColor.g, objectColor.b, objectColor.a);
					else
					{
						colorIndex = meshColorIndexes_[(h+1) * meshWidth_ + (w+1)] * paletteColorDim_;
						r = palette_[colorIndex] / 255.0f;
						g = palette_[colorIndex + 1] / 255.0f;
						b = palette_[colorIndex + 2] / 255.0f;
						glColor3f(r, g, b);
					}
					//glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);
					//
					if (useObjectColor) glColor4f(objectColor.r, objectColor.g, objectColor.b, objectColor.a);
					else
					{
						colorIndex = meshColorIndexes_[(h+1) * meshWidth_ + w] * paletteColorDim_;
						r = palette_[colorIndex] / 255.0f;
						g = palette_[colorIndex + 1] / 255.0f;
						b = palette_[colorIndex + 2] / 255.0f;
						glColor3f(r, g, b);
					}
					//glNormal3f(nx, ny, nz);
					glVertex3f(x4, y4, z4);
				}
		glEnd();
	glPopName();

	// pop original attributes
	glPopAttrib();  // GL_LIGHTING_BIT
}

void ColoredMeshShape::loadMesh()
{
	mesh_.reset(new float [meshWidth_ * meshHeight_]);
	meshColorIndexes_.reset(new unsigned char [meshWidth_ * meshHeight_]);
	palette_.reset(new unsigned char [paletteSize_ * paletteColorDim_]);

	if (!mesh_ || !meshColorIndexes_ || !palette_) return;

	{
#if defined(DEBUG) || defined(_DEBUG)
		std::wifstream stream(L"..\\data\\temp_value_320x240.txt");
#else
		std::ifstream stream("..\\data\\temp_value_320x240.txt");
#endif
		if (stream.is_open())
		{
			for (size_t i = 0; i < meshHeight_; ++i)
				for (size_t j = 0; j < meshWidth_; ++j)
					stream >> mesh_[i * meshWidth_ + j];

			stream.close();

			meshMinValue_ = *std::min_element(mesh_.get(), mesh_.get() + meshWidth_ * meshHeight_);
			meshMaxValue_ = *std::max_element(mesh_.get(), mesh_.get() + meshWidth_ * meshHeight_);
		}
		else
		{
			std::cout << "temperature data fail to be loaded !!!" << std::endl;
			return;
		}
	}

	{
#if defined(DEBUG) || defined(_DEBUG)
		std::wifstream stream(L"..\\data\\color_index_320x240.txt");
#else
		std::ifstream stream("..\\data\\color_index_320x240.txt");
#endif
		if (stream.is_open())
		{
			int ch;
			for (size_t i = 0; i < meshHeight_; ++i)
				for (size_t j = 0; j < meshWidth_; ++j)
				{
					stream >> ch;
					meshColorIndexes_[i * meshWidth_ + j] = (unsigned char)ch;
				}

			stream.close();
		}
		else
		{
			std::cout << "color index data fail to be loaded !!!" << std::endl;
			return;
		}
	}

	{
#if defined(DEBUG) || defined(_DEBUG)
		std::wifstream stream(L"..\\data\\rgb_palette.txt");
#else
		std::ifstream stream("..\\data\\rgb_palette.txt");
#endif
		if (stream.is_open())
		{
			int ch;
			for (size_t i = 0; i < paletteSize_; ++i)
				for (size_t j = 0; j < paletteColorDim_; ++j)
				{
					stream >> ch;
					palette_[i * paletteColorDim_ + j] = (unsigned char)ch;
				}

			stream.close();
		}
		else
		{
			std::cout << "RGB palette_ fails to be loaded !!!" << std::endl;
			return;
		}
	}
}

void ColoredMeshShape::calculateNormal(const float vx1, const float vy1, const float vz1, const float vx2, const float vy2, const float vz2, float &nx, float &ny, float &nz) const
{
	nx = vy1 * vz2 - vz1 * vy2;
	ny = vz1 * vx2 - vx1 * vz2;
	nz = vx1 * vy2 - vy1 * vx2;

	const float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
	nx /= norm;
	ny /= norm;
	nz /= norm;
}

//-----------------------------------------------------------------------------
//

TexturedMeshShape::TexturedMeshShape()
: base_type(),
  texWidth_(512), texHeight_(256)
{
}

TexturedMeshShape::~TexturedMeshShape()
{
	glDeleteTextures(texCount_, textureObjs_);
}

void TexturedMeshShape::draw() const
{
	drawTexturedMesh();
	//drawTexture();
	//drawMesh();
}

bool TexturedMeshShape::createDisplayList()
{
	glDeleteTextures(texCount_, textureObjs_);
	glGenTextures(texCount_, textureObjs_);
	createTexture();

	return base_type::createDisplayList();
}

void TexturedMeshShape::createTexture()
{
	//if (!meshColorIndexes_ || !palette_ || !glIsTexture(textureObjs_[0])) return;  // not working: i don't know why
	if (!meshColorIndexes_ || !palette_) return;

	boost::scoped_array<unsigned char> pixels(new unsigned char [texWidth_ * texHeight_ * paletteColorDim_]);
	if (!pixels) return;

	for (size_t h = 0; h < texHeight_; ++h)
		for (size_t w = 0; w < texWidth_; ++w)
		{
			const size_t w1 = size_t((float)w / (float)texWidth_ * meshWidth_);
			const size_t h1 = size_t((float)h / (float)texHeight_ * meshHeight_);
			const size_t colorIndex = meshColorIndexes_[h1 * meshWidth_ + w1] * paletteColorDim_;

			const size_t idx = (h * texWidth_ + w) * paletteColorDim_;
			pixels[idx] = palette_[colorIndex];
			pixels[idx + 1] = palette_[colorIndex + 1];
			pixels[idx + 2] = palette_[colorIndex + 2];
		}

	// set a texture object
	glBindTexture(GL_TEXTURE_2D, textureObjs_[0]);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, (GLsizei)texWidth_, (GLsizei)texHeight_, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.get());
	pixels.reset();

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	//const GLfloat texBlendColor[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
	//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_BLEND);
	//glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, texBlendColor);

	// restore to the unnamed default texture
	glBindTexture(GL_TEXTURE_2D, 0);
}

void TexturedMeshShape::drawTexturedMesh() const
{
	if (!mesh_ || !glIsTexture(textureObjs_[0])) return;

	// save states
	glPushAttrib(GL_TEXTURE_BIT);

	// set attributes
	glEnable(GL_TEXTURE_2D);

	// set a texture object
	glBindTexture(GL_TEXTURE_2D, textureObjs_[0]);

	const GLuint id = reinterpret_cast<GLuint>(this);
	glPushName(id);
		bool useObjectColor = false;
		swl::ObjectPickerMgr::color_type objectColor;
		if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(id))
		{
			objectColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
			useObjectColor = true;
		}
		else if (swl::ObjectPickerMgr::getInstance().isPickedObject(id))
		{
			objectColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
			useObjectColor = true;
		}

		float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, temp1, temp2, temp3, temp4;
		float nx, ny, nz;
		glBegin(GL_QUADS);
			for (size_t h = 0; h < meshHeight_ - 1; ++h)
				for (size_t w = 0; w < meshWidth_ - 1; ++w)
				{
					x1 = float(w) + xOffset_;
					x2 = float(w+1) + xOffset_;
					x3 = float(w+1) + xOffset_;
					x4 = float(w) + xOffset_;
#if 0
					y1 = float(h) + yOffset_;
					y2 = float(h) + yOffset_;
					y3 = float(h+1) + yOffset_;
					y4 = float(h+1) + yOffset_;
#else
					// flipped image
					y1 = float(meshHeight_ - h) + yOffset_;
					y2 = float(meshHeight_ - h) + yOffset_;
					y3 = float(meshHeight_ - (h+1)) + yOffset_;
					y4 = float(meshHeight_ - (h+1)) + yOffset_;
#endif

					temp1 = mesh_[h * meshWidth_ + w];
					z1 = temp1 * zScaleFactor_ + zOffset_;
					temp2 = mesh_[h * meshWidth_ + (w+1)];
					z2 = temp2 * zScaleFactor_ + zOffset_;
					temp3 = mesh_[(h+1) * meshWidth_ + (w+1)];
					z3 = temp3 * zScaleFactor_ + zOffset_;
					temp4 = mesh_[(h+1) * meshWidth_ + w];
					z4 = temp4 * zScaleFactor_ + zOffset_;

					//glEdgeFlag(GL_TRUE);

					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);
					glColor4f(red(), green(), blue(), alpha());

					//
					glNormal3f(nx, ny, nz);
					glTexCoord2f(float(w) / float(meshWidth_ - 1), float(h) / float(meshHeight_ - 1));
					glVertex3f(x1, y1, z1);
					//
					glNormal3f(nx, ny, nz);
					glTexCoord2f(float(w+1) / float(meshWidth_ - 1), float(h) / float(meshHeight_ - 1));
					glVertex3f(x2, y2, z2);
					//
					glNormal3f(nx, ny, nz);
					glTexCoord2f(float(w+1) / float(meshWidth_ - 1), float(h+1) / float(meshHeight_ - 1));
					glVertex3f(x3, y3, z3);
					//
					glNormal3f(nx, ny, nz);
					glTexCoord2f(float(w) / float(meshWidth_ - 1), float(h+1) / float(meshHeight_ - 1));
					glVertex3f(x4, y4, z4);
				}
		glEnd();
	glPopName();

	// restore to the unnamed default texture
	//glBindTexture(GL_TEXTURE_2D, 0);

	// pop original attributes
	glPopAttrib();  // GL_TEXTURE_BIT
}

void TexturedMeshShape::drawTexture() const
{
	if (!meshColorIndexes_ || !palette_) return;

	const boost::scoped_array<unsigned char> pixels(new unsigned char [meshWidth_ * meshHeight_ * paletteColorDim_]);
	if (!pixels) return;

	for (size_t h = 0; h < meshHeight_; ++h)
		for (size_t w = 0; w < meshWidth_; ++w)
		{
			const size_t idx = (h * meshWidth_ + w) * paletteColorDim_;
			const size_t colorIndex = meshColorIndexes_[h * meshWidth_ + w] * paletteColorDim_;
			pixels[idx] = palette_[colorIndex];
			pixels[idx + 1] = palette_[colorIndex + 1];
			pixels[idx + 2] = palette_[colorIndex + 2];
		}

	//glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	//glRasterPos2i(0, 0);
	glRasterPos3i(0, 0, 0);
	glDrawPixels((GLsizei)meshWidth_, (GLsizei)meshHeight_, GL_RGB, GL_UNSIGNED_BYTE, pixels.get());
}

void TexturedMeshShape::drawMesh() const
{
	if (!mesh_ || !meshColorIndexes_ || !palette_) return;

	float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4, temp1, temp2, temp3, temp4;
	float nx, ny, nz;
	glBegin(GL_QUADS);
		for (size_t h = 0; h < meshHeight_ - 1; ++h)
			for (size_t w = 0; w < meshWidth_ - 1; ++w)
			{
				x1 = float(w) + xOffset_;
				x2 = float(w+1) + xOffset_;
				x3 = float(w+1) + xOffset_;
				x4 = float(w) + xOffset_;
#if 0
				y1 = float(h) + yOffset_;
				y2 = float(h) + yOffset_;
				y3 = float(h+1) + yOffset_;
				y4 = float(h+1) + yOffset_;
#else
				// flipped image
				y1 = float(meshHeight_ - h) + yOffset_;
				y2 = float(meshHeight_ - h) + yOffset_;
				y3 = float(meshHeight_ - (h+1)) + yOffset_;
				y4 = float(meshHeight_ - (h+1)) + yOffset_;
#endif

				temp1 = mesh_[h * meshWidth_ + w];
				z1 = temp1 * zScaleFactor_ + zOffset_;
				temp2 = mesh_[h * meshWidth_ + (w+1)];
				z2 = temp2 * zScaleFactor_ + zOffset_;
				temp3 = mesh_[(h+1) * meshWidth_ + (w+1)];
				z3 = temp3 * zScaleFactor_ + zOffset_;
				temp4 = mesh_[(h+1) * meshWidth_ + w];
				z4 = temp4 * zScaleFactor_ + zOffset_;

				//glEdgeFlag(GL_TRUE);

				calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);
				glColor4f(red(), green(), blue(), alpha());

				//
				glNormal3f(nx, ny, nz);
				glVertex3f(x1, y1, z1);
				//
				glNormal3f(nx, ny, nz);
				glVertex3f(x2, y2, z2);
				//
				glNormal3f(nx, ny, nz);
				glVertex3f(x3, y3, z3);
				//
				glNormal3f(nx, ny, nz);
				glVertex3f(x4, y4, z4);
			}
	glEnd();
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

	GLint oldMatrixMode = 0;
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

//-----------------------------------------------------------------------------
//

void FloorShape::draw() const
{
#if 0
	const float minXBound = -500.0f, maxXBound = 500.0f;
	const float minYBound = -500.0f, maxYBound = 500.0f;
	const float minZBound = -500.0f, maxZBound = 500.0f;
#else
	const float minXBound = 0.0f, maxXBound = 320.0f;
	const float minYBound = 0.0f, maxYBound = 240.0f;
	const float minZBound = 0.0f, maxZBound = 40.0f;
#endif
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

		GLfloat oldLineWidth = 1.0f;
		glGetFloatv(GL_LINE_WIDTH, &oldLineWidth);
		glLineWidth(1.0f);

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

		glLineWidth(oldLineWidth);

		if (isLighting) glEnable(GL_LIGHTING);
		if (isDepthTest) glEnable(GL_DEPTH_TEST);
	}
}

bool FloorShape::createDisplayList()
{
	throw std::runtime_error("OpenGL display list is not used");
}

void FloorShape::callDisplayList() const
{
	throw std::runtime_error("OpenGL display list is not used");
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

	GLint oldMatrixMode = 0;
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

//-----------------------------------------------------------------------------
//

void CoordinateFrameShape::draw() const
{
	const boost::shared_ptr<view_type::camera_type> &camera = view_.topCamera();
	if (NULL == camera.get()) return;
 
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
	throw std::runtime_error("OpenGL display list is not used");
}

void CoordinateFrameShape::callDisplayList() const
{
	throw std::runtime_error("OpenGL display list is not used");
}

void CoordinateFrameShape::processToPick(const int x, const int y, const int width, const int height) const
{
	const boost::shared_ptr<view_type::context_type> &context = view_.topContext();
	const boost::shared_ptr<view_type::camera_type> &camera = view_.topCamera();
	if (!context || !camera) return;

	const swl::Region2<int> &oldViewport = camera->getViewport();
	const swl::Region2<double> &oldViewRegion = camera->getViewRegion();

	const int dX = int(oldViewport.getWidth() * 0.10);
	const int dY = int(oldViewport.getHeight() * 0.10);
	const int size = std::max(std::max(dX, dY), 100);

	camera->setViewport(swl::Region2<int>(oldViewport.left, oldViewport.bottom, size, size));
	camera->setViewRegion(static_cast<swl::ViewCamera2 *>(camera.get())->getViewBound());
	const swl::Region2<double> &currViewRegion = camera->getCurrentViewRegion();

	// save states
	glDisable(GL_DEPTH_TEST);

	//double modelviewMatrix[16];
	double projectionMatrix[16];
	int viewport[4];
	//glGetDoublev(GL_MODELVIEW_MATRIX, modelviewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, projectionMatrix);
	glGetIntegerv(GL_VIEWPORT, viewport);

	// set projection matrix
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	//gluPickMatrix(x, viewport[3] - y, width, height, viewport);
	gluPickMatrix(x, oldViewport.getHeight() - y, width, height, viewport);

	// need to load current projection matrix
	glMultMatrixd(projectionMatrix);

	//
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
/*
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		camera->lookAt();
*/
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

	// pop projection matrix
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	// restore states
	camera->setViewRegion(oldViewRegion);
	camera->setViewport(oldViewport);
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

	const GLuint id = reinterpret_cast<GLuint>(this);
	glPushName(id);
	for (int i = 0; i < 3; ++i)
	{
		if (0 == order[i])
		{
			const GLuint xId = id + 0;
			glPushName(xId);
				// x axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(xId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(xId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else glColor3f(1.0f, 0.0f, 0.0f);
				glPushMatrix();
					glRotated(90.0, 0.0, 1.0, 0.0);
					gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
					glTranslated(0.0, 0.0, size);
					gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
				glPopMatrix();

#if defined(_UNICODE) || defined(UNICODE)
				swl::WglFont::getInstance().drawText(height, 0.0f, 0.0f, L"X");
#else
     			swl::WglFont::getInstance().drawText(height, 0.0f, 0.0f, "X");
#endif
			glPopName();
		}
		else if (1 == order[i])
		{
			const GLuint yId = id + 1;
			glPushName(yId);
				// y axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(yId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(yId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else glColor3f(0.0f, 1.0f, 0.0f);
				glPushMatrix();
					glRotated(-90.0, 1.0, 0.0, 0.0);
					gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
					glTranslated(0.0, 0.0, size);
					gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
				glPopMatrix();

#if defined(_UNICODE) || defined(UNICODE)
				swl::WglFont::getInstance().drawText(0.0f, height, 0.0f, L"Y");
#else
				swl::WglFont::getInstance().drawText(0.0f, height, 0.0f, "Y");
#endif
			glPopName();
		}
		else if (2 == order[i])
		{
			const GLuint zId = id + 2;
			glPushName(zId);
				// z axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(zId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(zId))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else glColor3f(0.0f, 0.0f, 1.0f);	
				glPushMatrix();
					gluCylinder(obj, radius, radius, size, 12, 1); // obj, base, top, height 
					glTranslated(0.0, 0.0, size);
					gluCylinder(obj, coneRadius, 0.0, coneHeight, 12, 1);
				glPopMatrix();

#if defined(_UNICODE) || defined(UNICODE)
				swl::WglFont::getInstance().drawText(0.0f, 0.0f, height, L"Z");
#else
				swl::WglFont::getInstance().drawText(0.0f, 0.0f, height, "Z");
#endif
			glPopName();
		}
	}
	glPopName();
 
	gluDeleteQuadric(obj);
}
