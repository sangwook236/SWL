// WglViewTestView.cpp : implementation of the CWglViewTestView class
//

#include "stdafx.h"
#include "swl/Config.h"
#include "WglViewTest.h"

#include "WglViewTestDoc.h"
#include "WglViewTestView.h"

#include "ViewStateMachine.h"
#include "ViewEventHandler.h"
#include "swl/winview/WglDoubleBufferedContext.h"
#include "swl/winview/WglBitmapBufferedContext.h"
#include "swl/winview/WglPrintContext.h"
#include "swl/winview/WglViewPrintApi.h"
#include "swl/winview/WglViewCaptureApi.h"
#include "swl/winview/WglFont.h"
#include "swl/glutil/GLCamera.h"
#include "swl/view/MouseEvent.h"
#include "swl/view/KeyEvent.h"
#include "swl/graphics/ShapeSceneNode.h"
#include "swl/graphics/ObjectPickerMgr.h"
#include "swl/math/MathConstant.h"
#include <boost/smart_ptr.hpp>
#include <boost/multi_array.hpp>
#include <GL/glut.h>
#include <limits>
#include <iostream>
#include <fstream>
#include <cassert>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#if defined(max)
#undef max
#endif
#if defined(min)
#undef min
#endif

#define __USE_OPENGL_DISPLAY_LIST 1


namespace {

void drawCube()
{
	const GLfloat len = 500.0f;

	glPushMatrix();
		glTranslatef(-len * 0.5f, -len * 0.5f, -len * 0.5f);
		glBegin(GL_QUADS);
			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(0.0f, len, 0.0f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(len, len, 0.0f);
			glNormal3f(0.0f, 0.0f, -1.0f);
			glVertex3f(len, 0.0f, 0.0f);

			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(0.0f, 0.0f, len);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(len, 0.0f, len);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(len, len, len);
			glNormal3f(0.0f, 0.0f, 1.0f);
			glVertex3f(0.0f, len, len);

			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(len, 0.0f, 0.0f);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(len, 0.0f, len);
			glNormal3f(0.0f, -1.0f, 0.0f);
			glVertex3f(0.0f, 0.0f, len);

			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(0.0f, len, 0.0f);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(0.0f, len, len);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(len, len, len);
			glNormal3f(0.0f, 1.0f, 0.0f);
			glVertex3f(len, len, 0.0f);

			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, 0.0f, len);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, len, len);
			glNormal3f(-1.0f, 0.0f, 0.0f);
			glVertex3f(0.0f, len, 0.0f);

			glColor3f(1.0f, 0.0f, 0.0f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(len, 0.0f, 0.0f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(len, len, 0.0f);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(len, len, len);
			glNormal3f(1.0f, 0.0f, 0.0f);
			glVertex3f(len, 0.0f, len);
		glEnd();
	glPopMatrix();
}

typedef boost::multi_array<float, 2> mesh_array_type;
boost::scoped_ptr<mesh_array_type> mesh;
int mesh_row = 0, mesh_col = 0;
float mesh_z_min = std::numeric_limits<float>::max(), mesh_z_max = 0.0f;
const float mesh_max_color_r = 1.0f, mesh_max_color_g = 1.0f, mesh_max_color_b = 0.0f;
const float mesh_min_color_r = 0.5f, mesh_min_color_g = 0.5f, mesh_min_color_b = 0.0f;

void loadMesh()
{
	const std::string filename("./data/mesh.txt");

	std::ifstream stream(filename.c_str());
	stream >> mesh_row >> mesh_col;

	mesh.reset(new mesh_array_type(boost::extents[mesh_row][mesh_col]));

	float dat;
	for (int i = 0; i < mesh_row; ++i)
		for (int j = 0; j < mesh_col; ++j)
		{
			stream >> dat;
			(*mesh)[i][j] = dat;

			if (dat < mesh_z_min)
				mesh_z_min = dat;
			if (dat > mesh_z_max)
				mesh_z_max = dat;
		}
}

void calculateNormal(const float vx1, const float vy1, const float vz1, const float vx2, const float vy2, const float vz2, float &nx, float &ny, float &nz)
{
	nx = vy1 * vz2 - vz1 * vy2;
	ny = vz1 * vx2 - vx1 * vz2;
	nz = vx1 * vy2 - vy1 * vx2;

	const float norm = std::sqrt(nx*nx + ny*ny + nz*nz);
	nx /= norm;
	ny /= norm;
	nz /= norm;
}

void drawMesh()
{
	if (mesh)
	{
		const float factor = 50.0f;

		const GLfloat material_none[] = { 0.0f, 0.0f, 0.0f, 1.0f };
		const GLfloat material_half[] = { 0.5f, 0.5f, 0.5f, 1.0f };
		const GLfloat material_white[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		const GLfloat shininess_none[] = { 0.0f };
		GLfloat material_diffuse[] = { 0.0f, 0.0f, 0.0f, 1.0f };

		float nx, ny, nz;
		float r, g, b;
		float ratio;

		GLenum polygonFacing = GL_FRONT_AND_BACK;
#if 0
		//glMaterialfv(polygonFacing, GL_AMBIENT, material_half);
		//glMaterialfv(polygonFacing, GL_SPECULAR, material_none);
		//glMaterialfv(polygonFacing, GL_SHININESS, shininess_none);
		//glMaterialfv(polygonFacing, GL_EMISSION, material_none);

		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		glBegin(GL_TRIANGLES);
			for (int i = 0; i < mesh_row - 1; ++i)
				for (int j = 0; j < mesh_col - 1; ++j)
				{
					x1 = float(i) * factor;
					y1 = float(j) * factor;
					z1 = (*mesh)[i][j] * factor;
					x2 = float(i+1) * factor;
					y2 = float(j) * factor;
					z2 = (*mesh)[i+1][j] * factor;
					x3 = float(i) * factor;
					y3 = float(j+1) * factor;
					z3 = (*mesh)[i][j+1] * factor;
					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					ratio = (z1 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					ratio = (z2 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					ratio = (z3 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);

					//
					x1 = float(i+1) * factor;
					y1 = float(j+1) * factor;
					z1 = (*mesh)[i+1][j+1] * factor;
					x2 = float(i) * factor;
					y2 = float(j+1) * factor;
					z2 = (*mesh)[i][j+1] * factor;
					x3 = float(i+1) * factor;
					y3 = float(j) * factor;
					z3 = (*mesh)[i+1][j] * factor;
					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					ratio = (z1 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					ratio = (z2 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					ratio = (z3 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);
				}
		glEnd();
#elif 0
		//glMaterialfv(polygonFacing, GL_AMBIENT, material_half);
		//glMaterialfv(polygonFacing, GL_SPECULAR, material_none);
		//glMaterialfv(polygonFacing, GL_SHININESS, shininess_none);
		//glMaterialfv(polygonFacing, GL_EMISSION, material_none);

		float x1, y1, z1, x2, y2, z2, x3, y3, z3;
		glBegin(GL_TRIANGLES);
			for (int i = 0; i < mesh_row - 1; ++i)
				for (int j = 0; j < mesh_col - 1; ++j)
				{
					x1 = float(i) * factor;
					y1 = float(j) * factor;
					z1 = (*mesh)[i][j] * factor;
					x2 = float(i+1) * factor;
					y2 = float(j) * factor;
					z2 = (*mesh)[i+1][j] * factor;
					x3 = float(i+1) * factor;
					y3 = float(j+1) * factor;
					z3 = (*mesh)[i+1][j+1] * factor;
					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					ratio = (z1 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					ratio = (z2 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					ratio = (z3 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);

					//
					x1 = float(i) * factor;
					y1 = float(j) * factor;
					z1 = (*mesh)[i][j] * factor;
					x2 = float(i+1) * factor;
					y2 = float(j+1) * factor;
					z2 = (*mesh)[i+1][j+1] * factor;
					x3 = float(i) * factor;
					y3 = float(j+1) * factor;
					z3 = (*mesh)[i][j+1] * factor;
					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					ratio = (z1 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					ratio = (z2 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					ratio = (z3 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);
				}
		glEnd();
#else
		//glMaterialfv(polygonFacing, GL_AMBIENT, material_half);
		//glMaterialfv(polygonFacing, GL_SPECULAR, material_none);
		//glMaterialfv(polygonFacing, GL_SHININESS, shininess_none);
		//glMaterialfv(polygonFacing, GL_EMISSION, material_none);

		float x1, y1, z1, x2, y2, z2, x3, y3, z3, x4, y4, z4;
		glBegin(GL_QUADS);
			for (int i = 0; i < mesh_row - 1; ++i)
				for (int j = 0; j < mesh_col - 1; ++j)
				{
					x1 = float(i) * factor;
					y1 = float(j) * factor;
					z1 = (*mesh)[i][j] * factor;
					x2 = float(i+1) * factor;
					y2 = float(j) * factor;
					z2 = (*mesh)[i+1][j] * factor;
					x3 = float(i+1) * factor;
					y3 = float(j+1) * factor;
					z3 = (*mesh)[i+1][j+1] * factor;
					x4 = float(i) * factor;
					y4 = float(j+1) * factor;
					z4 = (*mesh)[i][j+1] * factor;
					calculateNormal(x2 - x1, y2 - y1, z2 - z1, x3 - x1, y3 - y1, z3 - z1, nx, ny, nz);

					//
					ratio = (z1 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					//glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x1, y1, z1);
					//
					ratio = (z2 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					//glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x2, y2, z2);
					//
					ratio = (z3 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					//glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x3, y3, z3);
					//
					ratio = (z4 / factor - mesh_z_min) / (mesh_z_max - mesh_z_min);
					r = mesh_min_color_r + (mesh_max_color_r - mesh_min_color_r) * ratio;
					g = mesh_min_color_g + (mesh_max_color_g - mesh_min_color_g) * ratio;
					b = mesh_min_color_b + (mesh_max_color_b - mesh_min_color_b) * ratio;
					//glEdgeFlag(GL_TRUE);
					glColor3f(r, g, b);
					//material_diffuse[0] = r;  material_diffuse[1] = g;  material_diffuse[2] = b;
					//glMaterialfv(polygonFacing, GL_DIFFUSE, material_diffuse);
					glNormal3f(nx, ny, nz);
					glVertex3f(x4, y4, z4);
				}
		glEnd();
#endif
	}
}

}  // unnamed namespace

// CWglViewTestView

IMPLEMENT_DYNCREATE(CWglViewTestView, CView)

BEGIN_MESSAGE_MAP(CWglViewTestView, CView)
	// Standard printing commands
	ON_COMMAND(ID_FILE_PRINT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_DIRECT, &CView::OnFilePrint)
	ON_COMMAND(ID_FILE_PRINT_PREVIEW, &CView::OnFilePrintPreview)
	ON_WM_DESTROY()
	ON_WM_PAINT()
	ON_WM_SIZE()
	ON_WM_LBUTTONDOWN()
	ON_WM_LBUTTONUP()
	ON_WM_LBUTTONDBLCLK()
	ON_WM_MBUTTONDOWN()
	ON_WM_MBUTTONUP()
	ON_WM_MBUTTONDBLCLK()
	ON_WM_RBUTTONDOWN()
	ON_WM_RBUTTONUP()
	ON_WM_RBUTTONDBLCLK()
	ON_WM_MOUSEMOVE()
	ON_WM_MOUSEWHEEL()
	ON_WM_KEYDOWN()
	ON_WM_KEYUP()
	ON_WM_CHAR()
	ON_COMMAND(ID_VIEWHANDLING_PAN, &CWglViewTestView::OnViewhandlingPan)
	ON_COMMAND(ID_VIEWHANDLING_ROTATE, &CWglViewTestView::OnViewhandlingRotate)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMREGION, &CWglViewTestView::OnViewhandlingZoomregion)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMALL, &CWglViewTestView::OnViewhandlingZoomall)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMIN, &CWglViewTestView::OnViewhandlingZoomin)
	ON_COMMAND(ID_VIEWHANDLING_ZOOMOUT, &CWglViewTestView::OnViewhandlingZoomout)
	ON_COMMAND(ID_VIEWHANDLING_PICKOBJECT, &CWglViewTestView::OnViewhandlingPickobject)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_PAN, &CWglViewTestView::OnUpdateViewhandlingPan)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ROTATE, &CWglViewTestView::OnUpdateViewhandlingRotate)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMREGION, &CWglViewTestView::OnUpdateViewhandlingZoomregion)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMALL, &CWglViewTestView::OnUpdateViewhandlingZoomall)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMIN, &CWglViewTestView::OnUpdateViewhandlingZoomin)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_ZOOMOUT, &CWglViewTestView::OnUpdateViewhandlingZoomout)
	ON_UPDATE_COMMAND_UI(ID_VIEWHANDLING_PICKOBJECT, &CWglViewTestView::OnUpdateViewhandlingPickobject)
	ON_COMMAND(ID_PRINTANDCAPTURE_PRINTVIEWUSINGGDI, &CWglViewTestView::OnPrintandcapturePrintviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDI, &CWglViewTestView::OnPrintandcaptureCaptureviewusinggdi)
	ON_COMMAND(ID_PRINTANDCAPTURE_CAPTUREVIEWUSINGGDIPLUS, &CWglViewTestView::OnPrintandcaptureCaptureviewusinggdiplus)
	ON_COMMAND(ID_PRINTANDCAPTURE_COPYTOCLIPBOARD, &CWglViewTestView::OnPrintandcaptureCopytoclipboard)
	ON_COMMAND(ID_EDIT_COPY, &CWglViewTestView::OnEditCopy)
END_MESSAGE_MAP()

// CWglViewTestView construction/destruction

CWglViewTestView::CWglViewTestView()
: swl::WglViewBase(),
  viewStateFsm_(),
  displayListHandler_(MAX_OPENGL_DISPLAY_LIST_COUNT),
  isPerspective_(true), isWireFrame_(false),
  polygonFacing_(GL_FRONT_AND_BACK),
  isGradientBackgroundUsed_(true), isFloorShown_(true), isColorBarShown_(true), isCoordinateFrameShown_(true),
  isPrinting_(false), isPickingObject_(false)
{
#if 0
	topGradientBackgroundColor_[0] = 0.776f;
	topGradientBackgroundColor_[1] = 0.835f;
	topGradientBackgroundColor_[2] = 0.980f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.243f;
	bottomGradientBackgroundColor_[1] = 0.443f;
	bottomGradientBackgroundColor_[2] = 0.968f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#elif 0
	topGradientBackgroundColor_[0] = 0.780f;
	topGradientBackgroundColor_[1] = 0.988f;
	topGradientBackgroundColor_[2] = 0.910f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.302f;
	bottomGradientBackgroundColor_[1] = 0.969f;
	bottomGradientBackgroundColor_[2] = 0.712f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#else
	topGradientBackgroundColor_[0] = 0.812f;
	topGradientBackgroundColor_[1] = 0.847f;
	topGradientBackgroundColor_[2] = 0.863f;
	topGradientBackgroundColor_[3] = 1.0f;
	bottomGradientBackgroundColor_[0] = 0.384f;
	bottomGradientBackgroundColor_[1] = 0.467f;
	bottomGradientBackgroundColor_[2] = 0.510f;
	bottomGradientBackgroundColor_[3] = 1.0f;
#endif

	floorColor_[0] = 0.5f;
	floorColor_[1] = 0.5f;
	floorColor_[2] = 0.5f;
	floorColor_[3] = 0.5f;

	// TODO [check] >> for testing
	loadMesh();
}

CWglViewTestView::~CWglViewTestView()
{
}

BOOL CWglViewTestView::PreCreateWindow(CREATESTRUCT& cs)
{
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return CView::PreCreateWindow(cs);
}

// CWglViewTestView drawing

void CWglViewTestView::OnDraw(CDC* pDC)
{
	CWglViewTestDoc* pDoc = GetDocument();
	ASSERT_VALID(pDoc);
	if (!pDoc)
		return;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	if (pDC && pDC->IsPrinting())
	{
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!camera) return;

		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));
		const int oldMapMode = pDC->SetMapMode(MM_TEXT);

		// save view's states
		const bool isPrinting = isPrinting_;
		if (!isPrinting) isPrinting_ = true;

		//
		const double eps = 1.0e-20;

		const swl::Region2<int> rctPage(swl::Point2<int>(0, 0), swl::Point2<int>(pDC->GetDeviceCaps(HORZRES), pDC->GetDeviceCaps(VERTRES)));
		const swl::Region2<double> &currViewRegion = camera->getCurrentViewRegion();
		const double width = currViewRegion.getWidth() >= eps ? currViewRegion.getWidth() : 1.0;
		const double height = currViewRegion.getHeight() >= eps ? currViewRegion.getHeight() : 1.0;
		const double ratio = std::min(rctPage.getWidth() / width, rctPage.getHeight() / height);

		const double width0 = width * ratio, height0 = height * ratio;
		const int w0 = (int)std::floor(width0), h0 = (int)std::floor(height0);
		const int x0 = rctPage.left + (int)std::floor((rctPage.getWidth() - width0) * 0.5), y0 = rctPage.bottom + (int)std::floor((rctPage.getHeight() - height0) * 0.5);

		const boost::shared_ptr<context_type> &context = topContext();

		swl::WglPrintContext printContext(pDC->GetSafeHdc(), swl::Region2<int>(x0, y0, x0 + w0, y0 + h0));
		const std::auto_ptr<camera_type> printCamera(dynamic_cast<WglViewBase::camera_type *>(camera->cloneCamera()));

		const bool isDisplayListShared = !context ? false : printContext.shareDisplayList(*context);

		if (printCamera.get() && printContext.isActivated())
		{
#if defined(__USE_OPENGL_DISPLAY_LIST)
			const bool doesRecreateDisplayListUsed = !isDisplayListShared && isDisplayListUsed();
			// generate a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) generateDisplayListName(true);
#endif

			initializeView();
			printCamera->setViewRegion(camera->getCurrentViewRegion());
			printCamera->setViewport(0, 0, w0, h0);

#if defined(__USE_OPENGL_DISPLAY_LIST)
			// re-create a OpenGL display list
			if (doesRecreateDisplayListUsed) createDisplayList(true);
#endif

			renderScene(printContext, *printCamera);

#if defined(__USE_OPENGL_DISPLAY_LIST)
			// delete a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) deleteDisplayListName(true);
#endif
		}

		// restore view's states
		if (!isPrinting) isPrinting_ = false;

		pDC->SetMapMode(oldMapMode);
		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
	else
	{
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (!camera) return;

		// using a locally-created context
		if (useLocallyCreatedContext_)
		{
			CRect rect;
			GetClientRect(&rect);

			boost::scoped_ptr<context_type> context;
			if (1 == drawMode_)
				context.reset(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect));
			else if (2 == drawMode_)
				context.reset(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect));

			if (context && context->isActivated())
			{
				initializeView();
				camera->setViewport(0, 0, rect.Width(), rect.Height());
				renderScene(*context, *camera);
			}
		}
		else
		{
			const boost::shared_ptr<context_type> &context = topContext();
			if (context && context->isActivated())
				renderScene(*context, *camera);
		}
	}
}


// CWglViewTestView printing

BOOL CWglViewTestView::OnPreparePrinting(CPrintInfo* pInfo)
{
	// default preparation
	return DoPreparePrinting(pInfo);
}

void CWglViewTestView::OnBeginPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add extra initialization before printing
}

void CWglViewTestView::OnEndPrinting(CDC* /*pDC*/, CPrintInfo* /*pInfo*/)
{
	// TODO: add cleanup after printing
}


// CWglViewTestView diagnostics

#ifdef _DEBUG
void CWglViewTestView::AssertValid() const
{
	CView::AssertValid();
}

void CWglViewTestView::Dump(CDumpContext& dc) const
{
	CView::Dump(dc);
}

CWglViewTestDoc* CWglViewTestView::GetDocument() const // non-debug version is inline
{
	ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CWglViewTestDoc)));
	return (CWglViewTestDoc*)m_pDocument;
}
#endif //_DEBUG


// CWglViewTestView message handlers

void CWglViewTestView::OnInitialUpdate()
{
	CView::OnInitialUpdate();

	//
	CRect rect;
	GetClientRect(&rect);

	drawMode_ = 2;  // [1, 2]
	useLocallyCreatedContext_ = false;

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
/*
	viewController_.addMousePressHandler(swl::MousePressHandler());
	viewController_.addMouseReleaseHandler(swl::MouseReleaseHandler());
	viewController_.addMouseMoveHandler(swl::MouseMoveHandler());
	viewController_.addMouseWheelHandler(swl::MouseWheelHandler());
	viewController_.addMouseClickHandler(swl::MouseClickHandler());
	viewController_.addMouseDoubleClickHandler(swl::MouseDoubleClickHandler());
	viewController_.addKeyPressHandler(swl::KeyPressHandler());
	viewController_.addKeyReleaseHandler(swl::KeyReleaseHandler());
	viewController_.addKeyHitHandler(swl::KeyHitHandler());
*/
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	// create a context
	if (1 == drawMode_)
		// it is not working with clipboard in Windows.
		pushContext(boost::shared_ptr<context_type>(new swl::WglDoubleBufferedContext(GetSafeHwnd(), rect, false)));
	else if (2 == drawMode_)
		pushContext(boost::shared_ptr<context_type>(new swl::WglBitmapBufferedContext(GetSafeHwnd(), rect, false)));

	// create a camera
	pushCamera(boost::shared_ptr<camera_type>(new swl::GLCamera()));

	const boost::shared_ptr<context_type> &viewContext = topContext();
	const boost::shared_ptr<camera_type> &viewCamera = topCamera();

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state

	if (!useLocallyCreatedContext_ && !viewStateFsm_ && viewContext && viewCamera)
	{
		viewStateFsm_.reset(new swl::ViewStateMachine(*this, *viewContext, *viewCamera));
		if (viewStateFsm_) viewStateFsm_->initiate();
	}

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	// initialize a view
	if (viewContext)
	{
		// guard the context
		context_type::guard_type guard(*viewContext);

#if defined(__USE_OPENGL_DISPLAY_LIST)
		generateDisplayListName(true);
#endif

		// set the view
		initializeView();

		// set the camera
		if (viewCamera)
		{
			// set the size of viewing volume
			viewCamera->setObjectPosition(0.0, 0.0, 0.0, false);
			//viewCamera->setObjectPosition(110.0, 110.0, 150.0, false);
			viewCamera->setEyePose(0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, false);
			//viewCamera->setEyePosition(1000.0, 1000.0, 1000.0, false);
			viewCamera->setEyeDistance(8000.0, false);
			//viewCamera->setEyeDistance(1000.0, false);

			// (left, bottom, right, top) is set wrt a eye coordinates frame
			// (near, far) is the distances from the eye point(viewpoint) to the near & far clipping planes of viewing volume
			//viewCamera->setViewBound(-1600.0, -1100.0, 2400.0, 2900.0, 1.0, 20000.0);
			viewCamera->setViewBound(-1000.0, -1000.0, 1000.0, 1000.0, 4000.0, 12000.0);
			//viewCamera->setViewBound(-50.0, -50.0, 50.0, 50.0, 1.0, 2000.0);

			viewCamera->setViewport(0, 0, rect.Width(), rect.Height());
			
			viewCamera->setPerspective(isPerspective_);
		}

#if defined(__USE_OPENGL_DISPLAY_LIST)
		createDisplayList(true);
#endif

		raiseDrawEvent(true);
	}

	// using a locally-created context
	if (useLocallyCreatedContext_)
		popContext();
}

void CWglViewTestView::OnDestroy()
{
	CView::OnDestroy();

#if defined(__USE_OPENGL_DISPLAY_LIST)
	deleteDisplayListName(false);
#endif

	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: basic routine

	popContext();
	popCamera();
}

void CWglViewTestView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	// using a locally-created context
	if (useLocallyCreatedContext_)
		raiseDrawEvent(true);
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (context)
		{
			if (context->isOffScreenUsed())
			{
				//context_type::guard_type guard(*context);
				context->swapBuffer();
			}
			else raiseDrawEvent(false);
		}
	}

	// Do not call CView::OnPaint() for painting messages
}

void CWglViewTestView::OnSize(UINT nType, int cx, int cy)
{
	CView::OnSize(nType, cx, cy);

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: basic routine

	if (cx <= 0 || cy <= 0) return;
	resizeView(0, 0, cx, cy);
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::raiseDrawEvent(const bool isContextActivated)
{
	if (isContextActivated)
		OnDraw(0L);
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context || context->isDrawing())
			return false;

		context_type::guard_type guard(*context);
		OnDraw(0L);
	}

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::initializeView()
{
	// can we put this in the constructor?
	// specify black(0.0f, 0.0f, 0.0f, 0.0f) or white(1.0f, 1.0f, 1.0f, 1.0f) as clear color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	// specify the back of the buffer as clear depth
    glClearDepth(1.0f);
	// enable depth testing
    glEnable(GL_DEPTH_TEST);
	// the type of depth testing
	glDepthFunc(GL_LESS);

	// enable stencil testing
	//glEnable(GL_STENCIL_TEST);
	// the type of stencil testing
	//glStencilFunc(GL_ALWAYS, 0, 1);
	//
	//glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);

	//
	//glEnable(GL_BLEND);
	////glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//
	//GLfloat pointSizes[2] = { 0.0f, };
	//glGetFloatv(GL_POINT_SIZE_RANGE, pointSizes);
	//GLfloat lineGradularities[2] = { 0.0f, };
	//glGetFloatv(GL_LINE_WIDTH_GRANULARITY, lineGradularities);
	//GLfloat lineWidths[2] = { 0.0f, };
	//glGetFloatv(GL_LINE_WIDTH_RANGE, lineWidths);

	//glEnable(GL_POINT_SMOOTH);
	//glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	//glEnable(GL_LINE_SMOOTH);
	//glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	//glEnable(GL_POLYGON_SMOOTH);
	//glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	// really nice perspective calculations
	//glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// lighting
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHT1);

	// create light components
	const GLfloat ambientLight[] = { 0.1f, 0.1f, 0.1f, 1.0f };
	const GLfloat diffuseLight[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	const GLfloat specularLight[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	const GLfloat position0[] = { 0.2f, 0.2f, 1.0f, 0.0f };
	const GLfloat position1[] = { 0.2f, 0.2f, -1.0f, 0.0f };

	// assign created components to GL_LIGHT0
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT0, GL_POSITION, position0);
	glLightfv(GL_LIGHT1, GL_AMBIENT, ambientLight);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuseLight);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specularLight);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);

	// polygon winding
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	//glEnable(GL_CULL_FACE);

	// surface normal
	glEnable(GL_NORMALIZE);
	glEnable(GL_AUTO_NORMAL);

	// shading model
	//glShadeModel(GL_FLAT);
	glShadeModel(GL_SMOOTH);

	// color tracking
	glEnable(GL_COLOR_MATERIAL);
	// set material properties which will be assigned by glColor
	glColorMaterial(polygonFacing_, GL_AMBIENT_AND_DIFFUSE);

	// clipping
	//int maxClipPlanes = 0;
	//glGetIntegerv(GL_MAX_CLIP_PLANES, &maxClipPlanes);

	glPolygonMode(polygonFacing_, GL_FILL);

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::resizeView(const int x1, const int y1, const int x2, const int y2)
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	deleteDisplayListName(false);
#endif

	const boost::shared_ptr<context_type> &context = topContext();
	if (context && context->resize(x1, y1, x2, y2))
	{
		context_type::guard_type guard(*context);

#if defined(__USE_OPENGL_DISPLAY_LIST)
		generateDisplayListName(true);
#endif

		initializeView();
		const boost::shared_ptr<camera_type> &camera = topCamera();
		if (camera)
		{
			camera->setViewport(x1, y1, x2, y2);

#if defined(__USE_OPENGL_DISPLAY_LIST)
			createDisplayList(true);
#endif
		}

		raiseDrawEvent(true);
		return true;
	}
	else return false;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::doPrepareRendering(const context_type & /*context*/, const camera_type & /*camera*/)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::doRenderStockScene(const context_type & /*context*/, const camera_type & /*camera*/)
{
	if (isGradientBackgroundUsed_ && !isPrinting_ && !isPickingObject_) drawGradientBackground();
	if (isFloorShown_ && !isPickingObject_) drawFloor();

	return true;
}

//-------------------------------------------------------------------------
// This code is required for SWL.WglView: basic routine

bool CWglViewTestView::doRenderScene(const context_type & /*context*/, const camera_type & /*camera*/)
{
	glPushMatrix();
		drawMainContent();
	glPopMatrix();

	if (isColorBarShown_ && !isPickingObject_) drawColorBar();
	if (isCoordinateFrameShown_ && !isPrinting_) drawCoordinateFrame();

    return true;
}


bool CWglViewTestView::createDisplayList(const bool isContextActivated)
{
	HDC *dc = NULL;

	const boost::shared_ptr<context_type> &context = topContext();
	if (context)
	{
		//context_type::guard_type guard(*context);
		try
		{
			dc = boost::any_cast<HDC *>(context->getNativeContext());
		}
		catch (const boost::bad_any_cast &)
		{
		}
	}

	if (isContextActivated)
	{
		createDisplayLists();
		if (dc)
		{
			swl::WglFont::getInstance().setDeviceContext(*dc);
			swl::WglFont::getInstance().createDisplayList();
		}
	}
	else
	{
		if (context)
		{
			context_type::guard_type guard(*context);
			createDisplayLists();
			if (dc)
			{
				swl::WglFont::getInstance().setDeviceContext(*dc);
				swl::WglFont::getInstance().createDisplayList();
			}
		}
	}

	return true;
}

void CWglViewTestView::generateDisplayListName(const bool isContextActivated)
{
	if (isContextActivated)
	{
		displayListHandler_.pushDisplayList();
		swl::WglFont::getInstance().pushDisplayList();
	}
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (context)
		{
			context_type::guard_type guard(*context);
			displayListHandler_.pushDisplayList();
			swl::WglFont::getInstance().pushDisplayList();
		}
	}
}

void CWglViewTestView::deleteDisplayListName(const bool isContextActivated)
{
	if (isContextActivated)
	{
		displayListHandler_.popDisplayList();
		swl::WglFont::getInstance().popDisplayList();
	}
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (context)
		{
			context_type::guard_type guard(*context);
			displayListHandler_.popDisplayList();
			swl::WglFont::getInstance().popDisplayList();
		}
	}
}

bool CWglViewTestView::isDisplayListUsed() const
{
	return displayListHandler_.isDisplayListUsed() || swl::WglFont::getInstance().isDisplayListUsed();
}

void CWglViewTestView::createDisplayLists() const
{
	// the name base of OpenGL display list that is actually used
	const unsigned int currDisplayListNameBase = displayListHandler_.getDisplayListNameBase();
	if (0u == currDisplayListNameBase) return;

	// for main content
	glNewList(currDisplayListNameBase + DLN_MAIN_CONTENT, GL_COMPILE);
		drawMainContent(true);
	glEndList();
/*
	// for floor
	glNewList(currDisplayListNameBase + DLN_FLOOR, GL_COMPILE);
		drawFloor(true);
	glEndList();
*/
	// for gradient background
	glNewList(currDisplayListNameBase + DLN_GRADIENT_BACKGROUND, GL_COMPILE);
		drawGradientBackground(true);
	glEndList();

	// for color bar
	glNewList(currDisplayListNameBase + DLN_COLOR_BAR, GL_COMPILE);
		drawColorBar(true);
	glEndList();
/*
	// for coordinate frame
	glNewList(currDisplayListNameBase + DLN_COORDINATE_FRAME, GL_COMPILE);
		drawCoordinateFrame(true);
	glEndList();
*/
}

void CWglViewTestView::pickObject(const int x, const int y, const bool isTemporary /*= false*/)
{
	processToPickObject(x, y, 2, 2, isTemporary);
}

void CWglViewTestView::pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary /*= false*/)
{
	const swl::Region2<int> region(swl::Point2<int>(x1, y1), swl::Point2<int>(x2, y2));
	processToPickObject(region.getCenterX(), region.getCenterY(), region.getWidth() > 0 ? region.getWidth() : 1, region.getHeight() > 0 ? region.getHeight() : 1, isTemporary);
}

void CWglViewTestView::processToPickObject(const int x, const int y, const int width, const int height, const bool isTemporary /*= false*/)
{
	const boost::shared_ptr<context_type> &context = topContext();
	const boost::shared_ptr<camera_type> &camera = topCamera();
	if (!context || !camera) return;

	context_type::guard_type guard(*context);

	// save states
	GLint oldMatrixMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	glPushAttrib(GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT);

	// set attributes
	glDisable(GL_LIGHTING);
	glDepthFunc(GL_LEQUAL);

	// set selection buffer
	const GLsizei SELECT_BUFFER_SIZE = 64;
	GLuint selectBuffer[SELECT_BUFFER_SIZE] = { 0, };
	glSelectBuffer(SELECT_BUFFER_SIZE, selectBuffer);

	// change rendering mode
	glRenderMode(GL_SELECT);

	// initialize name stack
	glInitNames();
	//glPushName(PON_BASE_ENTRY);

	// render scene
	isPickingObject_ = true;
#if 0
	renderScene(*context, *camera);
#else
	processToPickMainContent(camera, x, y, width, height);
	if (isCoordinateFrameShown_) processToPickCoordinateFrame(camera, x, y, width, height);
#endif
	isPickingObject_ = false;

	// gather hit records
	const GLint hitCount = glRenderMode(GL_RENDER);

	// pop original attributes
	glPopAttrib();  // GL_LIGHTING_BIT | GL_DEPTH_BUFFER_BIT

	glMatrixMode(oldMatrixMode);

	// process hits
	const unsigned int pickedObj = hitCount > 0 ? processHits(hitCount, selectBuffer) : PON_BASE_ENTRY;
	if (isTemporary)
	{
		if (PON_BASE_ENTRY == pickedObj && swl::ObjectPickerMgr::getInstance().containTemporarilyPickedObject())
		{
			swl::ObjectPickerMgr::getInstance().clearAllTemporarilyPickedObjects();
			raiseDrawEvent(false);
		}
		else if (PON_BASE_ENTRY != pickedObj && !swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(pickedObj))
		{
			swl::ObjectPickerMgr::getInstance().clearAllTemporarilyPickedObjects();
			swl::ObjectPickerMgr::getInstance().addTemporarilyPickedObject(pickedObj);
			raiseDrawEvent(false);
		}
	}
	else
	{
		const bool isTemporarilyPickedObj = swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(pickedObj);
		swl::ObjectPickerMgr::getInstance().clearAllTemporarilyPickedObjects();

		if (PON_BASE_ENTRY == pickedObj && swl::ObjectPickerMgr::getInstance().containPickedObject())
		{
			swl::ObjectPickerMgr::getInstance().clearAllPickedObjects();
			raiseDrawEvent(false);
		}
		else if (PON_BASE_ENTRY != pickedObj && !swl::ObjectPickerMgr::getInstance().isPickedObject(pickedObj))
		{
			swl::ObjectPickerMgr::getInstance().clearAllPickedObjects();
			swl::ObjectPickerMgr::getInstance().addPickedObject(pickedObj);

			// TODO [add] >> process picked objects
			switch (pickedObj)
			{
			case PON_SPHERE:
				break;
			case PON_CUBE:
				break;
			case PON_X_AXIS:
				break;
			case PON_Y_AXIS:
				break;
			case PON_Z_AXIS:
				break;
			}

			if (!isTemporarilyPickedObj) raiseDrawEvent(false);
		}
	}
}

void CWglViewTestView::processToPickMainContent(const boost::shared_ptr<camera_type> &camera, const int x, const int y, const int width, const int height) const
{
	// set attributes
	glEnable(GL_DEPTH_TEST);
	//glDepthRange(0.0, 1.0);

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
	gluPickMatrix(x, viewport[3] - y, width, height, viewport);

	// need to load current projection matrix
	glMultMatrixd(projectionMatrix);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		camera->lookAt();

#if 0
		doRenderScene(*context, *camera);
#else
		drawMainContent();
#endif
	glPopMatrix();

	// pop projection matrix
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
}

void CWglViewTestView::processToPickCoordinateFrame(const boost::shared_ptr<camera_type> &camera, const int x, const int y, const int width, const int height) const
{
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
		glLoadIdentity();
		// 1. need to load current modelview matrix
		//   e.g.) glLoadMatrixd(modelviewMatrix);
		// 2. need to be thought of viewing transformation
		//   e.g.) camera->lookAt();
		camera->lookAt();

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

unsigned int CWglViewTestView::processHits(const int hitCount, const unsigned int *buffer) const
{
	const GLuint *ptr = (const GLuint *)buffer;

	GLuint selectedObj = PON_BASE_ENTRY;
	bool isCoordinateFramePicked = false;
	//float minZ = 1.0f;
	unsigned int minZ = 0xffffffff;
	for (int i = 0; i < hitCount; ++i)
	{
		// number of names for each hit.
		const GLuint nameCount = *ptr;
		++ptr;
		// min. window-coordinate z values of all vertices of the primitives that intersectd the viewing volume since the last recorded hit.
		////const float mnZ = float(*ptr) / 0x7fffffff;
		//const float mnZ = float(*ptr) / 0xffffffff;  // 2^32 - 1
		const unsigned int mnZ = *ptr;
		++ptr;
		// max. window-coordinate z values of all vertices of the primitives that intersectd the viewing volume since the last recorded hit
		////const float mxZ = float(*ptr) / 0x7fffffff;
		//const float mxZ = float(*ptr) / 0xffffffff;  // 2^32 - 1
		const unsigned int mxZ = *ptr;
		++ptr;

		if (0 == nameCount) continue;

		const GLuint currObj = *(ptr + nameCount - 1);
		if (isCoordinateFramePicked)
		{
			switch (currObj)
			{
			case PON_X_AXIS:
			case PON_Y_AXIS:
			case PON_Z_AXIS:
				if (mnZ < minZ)
				{
					minZ = mnZ;
					selectedObj = currObj;
				}
				break;
			}
		}
		else
		{
			switch (currObj)
			{
			case PON_X_AXIS:
			case PON_Y_AXIS:
			case PON_Z_AXIS:
				minZ = mnZ;
				selectedObj = currObj;
				isCoordinateFramePicked = true;
				break;
			default:
				if (mnZ < minZ)
				{
					minZ = mnZ;
					selectedObj = currObj;
				}
				break;
			}
		}

		const GLuint *ptr2 = ptr;
		//TRACE("***** the number of names for each hit: %d, min z: %f, max z: %f\n", nameCount, mnZ, mxZ);
		TRACE("***** the number of names for each hit: %d, min z: %d, max z: %d\n", nameCount, mnZ, mxZ);
		// the contents of the name stack
		TRACE("\tthe contents of the name stack: %d", *ptr2);
		++ptr2;
		for (GLuint j = 1; j < nameCount; ++j)
		{
			const GLint name = *ptr2;
			++ptr2;
			TRACE(" - %d", name);
		}
		TRACE("\n");

		ptr += nameCount;
	}

	TRACE("=====> The picked object: %d\n", selectedObj);
	return selectedObj;
}

void CWglViewTestView::dragObject(const int x1, const int y1, const int x2, const int y2)
{
	// FIXME [implement] >>
	throw std::runtime_error("Not yet implemented");
}

void CWglViewTestView::setPerspective(const bool isPerspective)
{
	if (isPerspective == isPerspective_) return;

	const boost::shared_ptr<context_type> &context = topContext();
	const boost::shared_ptr<camera_type> &camera = topCamera();
	if (context && camera)
	{
		isPerspective_ = isPerspective;

		context_type::guard_type guard(*context);
		camera->setPerspective(isPerspective_);

//#if defined(__USE_OPENGL_DISPLAY_LIST)
//		createDisplayList(true);
//#endif
	}
}

void CWglViewTestView::setWireFrame(const bool isWireFrame)
{
	if (isWireFrame == isWireFrame_) return;

	isWireFrame_ = isWireFrame;

#if defined(__USE_OPENGL_DISPLAY_LIST)
	createDisplayList(false);
#endif
}

void CWglViewTestView::drawMainContent(const bool doesCreateDisplayList /*= false*/) const
{
	//const bool isPickObjectState = viewStateFsm_ && NULL != viewStateFsm_->state_cast<const swl::PickObjectState *>();
	const bool isPickObjectState = swl::ObjectPickerMgr::getInstance().isPicking();

#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!doesCreateDisplayList && isDisplayListUsed() && !isPickObjectState)
	{
		glCallList(displayListHandler_.getDisplayListNameBase() + DLN_MAIN_CONTENT);
		return;
	}
#endif

#if 1
	// save states
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);

	glPolygonMode(polygonFacing_, isWireFrame_ ? GL_LINE : GL_FILL);
	//glPolygonMode(GL_FRONT, isWireFrame_ ? GL_LINE : GL_FILL);  // not working !!!

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

		glPushName(PON_SPHERE);
			// draw a sphere
			if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(PON_SPHERE))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else if (swl::ObjectPickerMgr::getInstance().isPickedObject(PON_SPHERE))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else glColor3f(1.0f, 0.0f, 0.0f);
			isWireFrame_ ? glutWireSphere(500.0, 20, 20) : glutSolidSphere(500.0, 20, 20);
		glPopName();

		// disables clipping planes
		glDisable(GL_CLIP_PLANE0);
		glDisable(GL_CLIP_PLANE1);
	glPopMatrix();

	glPushMatrix();
		glTranslatef(250.0f, -250.0f, 250.0f);

		glPushName(PON_CUBE);
			// draw a cube
			if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(PON_CUBE))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else if (swl::ObjectPickerMgr::getInstance().isPickedObject(PON_CUBE))
			{
				const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getPickedColor();
				glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
			}
			else glColor3f(0.5f, 0.5f, 1.0f);
			isWireFrame_ ? glutWireCube(500.0) : glutSolidCube(500.0);
		glPopName();
	glPopMatrix();

	// restore states
	//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
	glPolygonMode(polygonFacing_, oldPolygonMode[1]);
#endif

#if 0
	// save states
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);

	glPushMatrix();
		glPolygonMode(polygonFacing_, isWireFrame_ ? GL_LINE : GL_FILL);
		//glPolygonMode(GL_FRONT, isWireFrame_ ? GL_LINE : GL_FILL);  // not working !!!

		//drawCube();
		drawMesh();
	glPopMatrix();

	// restore states
	//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
	glPolygonMode(polygonFacing_, oldPolygonMode[1]);
#endif
}

void CWglViewTestView::drawGradientBackground(const bool doesCreateDisplayList /*= false*/) const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!doesCreateDisplayList && isDisplayListUsed())
	//if (!doesCreateDisplayList && isDisplayListUsed() && !isPickObjectState)
	{
		glCallList(displayListHandler_.getDisplayListNameBase() + DLN_GRADIENT_BACKGROUND);
		return;
	}
#endif

	// save states
	const GLboolean isLighting = glIsEnabled(GL_LIGHTING);
	if (isLighting) glDisable(GL_LIGHTING);
	const GLboolean isDepthTest = glIsEnabled(GL_DEPTH_TEST);
	if (isDepthTest) glDisable(GL_DEPTH_TEST);
	GLint oldPolygonMode[2];
	glGetIntegerv(GL_POLYGON_MODE, oldPolygonMode);
	if (GL_FILL != oldPolygonMode[1]) glPolygonMode(polygonFacing_, GL_FILL);

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
			glColor4fv(bottomGradientBackgroundColor_);
			glVertex3f(0.0f, 0.0f, 0.0f);
			glVertex3f(1.0f, 0.0f, 0.0f);

			glColor4fv(topGradientBackgroundColor_);
			glVertex3f(1.0f, 1.0f, 0.0f);
			glVertex3f(0.0f, 1.0f, 0.0f);
		glEnd();
	glPopMatrix();

	// restore modelview matrix
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// restore states
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

	if (GL_FILL != oldPolygonMode[1])
		//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
		glPolygonMode(polygonFacing_, oldPolygonMode[1]);
	if (isLighting) glEnable(GL_LIGHTING);
	if (isDepthTest) glEnable(GL_DEPTH_TEST);
}

void CWglViewTestView::drawFloor(const bool doesCreateDisplayList /*= false*/) const
{
/*
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!doesCreateDisplayList && isDisplayListUsed())
	//if (!doesCreateDisplayList && isDisplayListUsed() && !isPickObjectState)
	{
		glCallList(displayListHandler_.getDisplayListNameBase() + DLN_FLOOR);
		return;
	}
#endif
*/

	const float minXBound = -500.0f, maxXBound = 500.0f;
	const float minYBound = -500.0f, maxYBound = 500.0f;
	const float minZBound = -500.0f, maxZBound = 500.0f;
	const float angleThreshold = (float)std::cos(80.0 * swl::MathConstant::TO_RAD);
	const size_t lineCount = 5;
	const int lineStippleScaleFactor = 2;

	const boost::shared_ptr<camera_type> &camera = topCamera();
	if (camera)
	{
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
		if (GL_LINE != oldPolygonMode[1]) glPolygonMode(polygonFacing_, GL_LINE);

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
			glColor4fv(floorColor_);

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
		if (GL_LINE != oldPolygonMode[1])
			//glPolygonMode(oldPolygonMode[0], oldPolygonMode[1]);  // not working. don't know why.
			glPolygonMode(polygonFacing_, oldPolygonMode[1]);
		if (!isLineStipple) glDisable(GL_LINE_STIPPLE);

		glLineWidth(oldLineWidth);

		if (isLighting) glEnable(GL_LIGHTING);
		if (isDepthTest) glEnable(GL_DEPTH_TEST);
	}
}

void CWglViewTestView::drawColorBar(const bool doesCreateDisplayList /*= false*/) const
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!doesCreateDisplayList && isDisplayListUsed())
	//if (!doesCreateDisplayList && isDisplayListUsed() && !isPickObjectState)
	{
		glCallList(displayListHandler_.getDisplayListNameBase() + DLN_COLOR_BAR);
		return;
	}
#endif

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

void CWglViewTestView::drawCoordinateFrame(const bool doesCreateDisplayList /*= false*/) const
{
	//const bool isPickObjectState = viewStateFsm_ && NULL != viewStateFsm_->state_cast<const swl::PickObjectState *>();
	const bool isPickObjectState = swl::ObjectPickerMgr::getInstance().isPicking();

/*
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (!doesCreateDisplayList && isDisplayListUsed() && !isPickObjectState)
	{
		glCallList(displayListHandler_.getDisplayListNameBase() + DLN_COORDINATE_FRAME);
		return
	}
#endif
*/

	const boost::shared_ptr<camera_type> &camera = topCamera();
	if (!camera) return;
 
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

void CWglViewTestView::drawCoordinateFrame(const float height, const int order[]) const
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
			glPushName(PON_X_AXIS);
				// x axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(PON_X_AXIS))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(PON_X_AXIS))
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
			glPushName(PON_Y_AXIS);
				// y axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(PON_Y_AXIS))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(PON_Y_AXIS))
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
			glPushName(PON_Z_AXIS);
				// z axis
				if (swl::ObjectPickerMgr::getInstance().isPicking() && swl::ObjectPickerMgr::getInstance().isTemporarilyPickedObject(PON_Z_AXIS))
				{
					const swl::ObjectPickerMgr::color_type &pickedColor = swl::ObjectPickerMgr::getInstance().getTemporarilyPickedColor();
					glColor4f(pickedColor.r, pickedColor.g, pickedColor.b, pickedColor.a);
				}
				else if (swl::ObjectPickerMgr::getInstance().isPickedObject(PON_Z_AXIS))
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
 
	gluDeleteQuadric(obj);
}

void CWglViewTestView::drawClippingArea(const unsigned int clippingPlaneId, const double *clippingPlaneEqn) const
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

	isWireFrame_ ? glutWireSphere(500.0, 20, 20) : glutSolidSphere(500.0, 20, 20);

	// second pass: decrement stencil buffer value on front faces
	glStencilOp(GL_KEEP, GL_KEEP, GL_DECR);
	glCullFace(GL_BACK);  // render front faces only

	isWireFrame_ ? glutWireSphere(500.0, 20, 20) : glutSolidSphere(500.0, 20, 20);

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

void CWglViewTestView::OnLButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDown(nFlags, point);
}

void CWglViewTestView::OnLButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonUp(nFlags, point);
}

void CWglViewTestView::OnLButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));
	if (viewStateFsm_) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_LEFT, ckey));

	CView::OnLButtonDblClk(nFlags, point);
}

void CWglViewTestView::OnMButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDown(nFlags, point);
}

void CWglViewTestView::OnMButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonUp(nFlags, point);
}

void CWglViewTestView::OnMButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));
	if (viewStateFsm_) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_MIDDLE, ckey));

	CView::OnMButtonDblClk(nFlags, point);
}

void CWglViewTestView::OnRButtonDown(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	SetCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_) viewStateFsm_->pressMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDown(nFlags, point);
}

void CWglViewTestView::OnRButtonUp(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	ReleaseCapture();

	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_) viewStateFsm_->releaseMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonUp(nFlags, point);
}

void CWglViewTestView::OnRButtonDblClk(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));
	if (viewStateFsm_) viewStateFsm_->doubleClickMouse(swl::MouseEvent(point.x, point.y, swl::MouseEvent::BT_RIGHT, ckey));

	CView::OnRButtonDblClk(nFlags, point);
}

void CWglViewTestView::OnMouseMove(UINT nFlags, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EButton btn = (swl::MouseEvent::EButton)(
		((nFlags & MK_LBUTTON) == MK_LBUTTON ? swl::MouseEvent::BT_LEFT : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_MBUTTON) == MK_MBUTTON ? swl::MouseEvent::BT_MIDDLE : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_RBUTTON) == MK_RBUTTON ? swl::MouseEvent::BT_RIGHT : swl::MouseEvent::BT_NONE)
	);
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.moveMouse(swl::MouseEvent(point.x, point.y, btn, ckey));
	if (viewStateFsm_) viewStateFsm_->moveMouse(swl::MouseEvent(point.x, point.y, btn, ckey));

	CView::OnMouseMove(nFlags, point);
}

BOOL CWglViewTestView::OnMouseWheel(UINT nFlags, short zDelta, CPoint point)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::MouseEvent::EButton btn = (swl::MouseEvent::EButton)(
		((nFlags & MK_LBUTTON) == MK_LBUTTON ? swl::MouseEvent::BT_LEFT : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_MBUTTON) == MK_MBUTTON ? swl::MouseEvent::BT_MIDDLE : swl::MouseEvent::BT_NONE) |
		((nFlags & MK_RBUTTON) == MK_RBUTTON ? swl::MouseEvent::BT_RIGHT : swl::MouseEvent::BT_NONE)
	);
	const swl::MouseEvent::EControlKey ckey = (swl::MouseEvent::EControlKey)(
		((nFlags & MK_CONTROL) == MK_CONTROL ? swl::MouseEvent::CK_CTRL : swl::MouseEvent::CK_NONE) |
		((nFlags & MK_SHIFT) == MK_SHIFT ? swl::MouseEvent::CK_SHIFT : swl::MouseEvent::CK_NONE)
	);
	//viewController_.wheelMouse(swl::MouseEvent(point.x, point.y, btn, ckey, swl::MouseEvent::SC_VERTICAL, zDelta / WHEEL_DELTA));
	if (viewStateFsm_) viewStateFsm_->wheelMouse(swl::MouseEvent(point.x, point.y, btn, ckey, swl::MouseEvent::SC_VERTICAL, zDelta / WHEEL_DELTA));

	return CView::OnMouseWheel(nFlags, zDelta, point);
}

void CWglViewTestView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	//viewController_.pressKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_) viewStateFsm_->pressKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyDown(nChar, nRepCnt, nFlags);
}

void CWglViewTestView::OnKeyUp(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt));
	if (viewStateFsm_) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt));

	CView::OnKeyUp(nChar, nRepCnt, nFlags);
}

void CWglViewTestView::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: event handling
	const swl::KeyEvent::EControlKey ckey = ((nFlags >> 28) & 0x01) == 0x01 ? swl::KeyEvent::CK_ALT : swl::KeyEvent::CK_NONE;
	//viewController_.releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));
	if (viewStateFsm_) viewStateFsm_->releaseKey(swl::KeyEvent(nChar, nRepCnt, ckey));

	CView::OnChar(nChar, nRepCnt, nFlags);
}

void CWglViewTestView::OnViewhandlingPan()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtPan());
}

void CWglViewTestView::OnViewhandlingRotate()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtRotate());
}

void CWglViewTestView::OnViewhandlingZoomregion()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtZoomRegion());
}

void CWglViewTestView::OnViewhandlingZoomall()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtZoomAll());
}

void CWglViewTestView::OnViewhandlingZoomin()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtZoomIn());
}

void CWglViewTestView::OnViewhandlingZoomout()
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtZoomOut());
}

void CWglViewTestView::OnViewhandlingPickobject()
{
	const bool isRedrawn = swl::ObjectPickerMgr::getInstance().containPickedObject();

	//-------------------------------------------------------------------------
	// This code is required for SWL.WglView: view state
	if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtPickObject());
	//if (viewStateFsm_) viewStateFsm_->process_event(swl::EvtPickAndDragObject());

	if (isRedrawn) raiseDrawEvent(false);
}

void CWglViewTestView::OnUpdateViewhandlingPan(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PanState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingRotate(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::RotateState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingZoomregion(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomRegionState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingZoomall(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomAllState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingZoomin(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomInState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingZoomout(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::ZoomOutState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnUpdateViewhandlingPickobject(CCmdUI *pCmdUI)
{
	//-------------------------------------------------------------------------
	// This code is required for SWL.WinView: view state
	if (viewStateFsm_)
		pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PickObjectState *>() ? 1 : 0);
		//pCmdUI->SetCheck(viewStateFsm_->state_cast<const swl::PickAndDragObjectState *>() ? 1 : 0);
	else pCmdUI->SetCheck(0);
}

void CWglViewTestView::OnPrintandcapturePrintviewusinggdi()
{
	// initialize a PRINTDLG structure
	PRINTDLG pd;
	memset(&pd, 0, sizeof(pd));
	pd.lStructSize = sizeof(pd);
	pd.hwndOwner = GetSafeHwnd();
	pd.Flags = PD_RETURNDC | PD_DISABLEPRINTTOFILE;
	pd.hInstance = NULL;
	if (!PrintDlg(&pd)) return;
	if (!pd.hDC) return;

	//
	const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

	// each logical unit is mapped to one device pixel. Positive x is to the right. positive y is down.
	SetMapMode(pd.hDC, MM_TEXT);

	DOCINFO di;
	di.cbSize = sizeof(DOCINFO);
	di.lpszDocName = _T("OpenGL Print");
	di.lpszOutput = NULL;

	// start the print job
	StartDoc(pd.hDC, &di);
	StartPage(pd.hDC);

	//
#if 0
	// save view's states
	const bool isPrinting = isPrinting_;
	if (!isPrinting) isPrinting_ = true;

	if (!swl::printWglViewUsingGdi(*this, pd.hDC))
		AfxMessageBox(_T("fail to print a view"), MB_OK | MB_ICONSTOP);

	// restore view's states
	if (!isPrinting) isPrinting_ = false;
#else
	CDC *pDC = CDC::FromHandle(pd.hDC);
	if (pDC)
	{
		pDC->m_bPrinting = TRUE;;
		OnDraw(pDC);
	}
#endif

	// end the print job
	EndPage(pd.hDC);
	EndDoc(pd.hDC);
	DeleteDC(pd.hDC);

	DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
}

void CWglViewTestView::OnPrintandcaptureCaptureviewusinggdi()
{
	CFileDialog dlg(FALSE, _T("bmp"), _T("*.bmp"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, _T("BMP Files (*.bmp)|*.bmp||"), NULL);
	dlg.m_ofn.lpstrTitle = _T("Capture View As");
	if (dlg.DoModal() == IDOK)
	{
		const HCURSOR oldCursor = SetCursor(LoadCursor(0L, IDC_WAIT));

#if defined(_UNICODE) || defined(UNICODE)
		const std::wstring filePathName((wchar_t *)(LPCTSTR)dlg.GetPathName());
#else
		const std::string filePathName((char *)(LPCTSTR)dlg.GetPathName());
#endif
		if (!swl::captureWglViewUsingGdi(filePathName, *this, GetSafeHwnd()))
			AfxMessageBox(_T("Fail to capture a view"), MB_OK | MB_ICONSTOP);

		DeleteObject(SetCursor(oldCursor ? oldCursor : LoadCursor(0L, IDC_ARROW)));
	}
}

void CWglViewTestView::OnPrintandcaptureCaptureviewusinggdiplus()
{
	// FIXME [add] >>
	AfxMessageBox(_T("Not yet implemented"), MB_OK | MB_ICONSTOP);
}

void CWglViewTestView::OnPrintandcaptureCopytoclipboard()
{
	CClientDC dc(this);

	CDC memDC;
	memDC.CreateCompatibleDC(&dc);

	CRect rect;
	GetWindowRect(&rect);

	CBitmap bitmap;
	bitmap.CreateCompatibleBitmap(&dc, rect.Width(), rect.Height());

	CBitmap *oldBitmap = memDC.SelectObject(&bitmap);
	memDC.BitBlt(0, 0, rect.Width(), rect.Height(), &dc, 0, 0, SRCCOPY);

	// clipboard
	if (OpenClipboard())
	{
		EmptyClipboard();
		SetClipboardData(CF_BITMAP, bitmap.GetSafeHandle());
		CloseClipboard();
	}

	memDC.SelectObject(oldBitmap);
	bitmap.Detach();
}

void CWglViewTestView::OnEditCopy()
{
	OnPrintandcaptureCopytoclipboard();
}
