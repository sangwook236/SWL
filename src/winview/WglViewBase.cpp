#include "swl/winview/WglViewBase.h"
#include "swl/winview/WglContextBase.h"
#include "swl/oglview/OglCamera.h"
#include <GL/glut.h>
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class WglViewBase

void WglViewBase::renderScene(context_type &viewContext, camera_type &viewCamera)
{
#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			std::cerr << "OpenGL error at " << __LINE__ << " in " << __FILE__ << ": " << gluErrorString(glErrorCode) << std::endl;
	}
#endif

	int oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);

	{
		glPushMatrix();
			//
			glLoadIdentity();
			viewCamera.lookAt();

			//
			glPushMatrix();
				doPrepareRendering(viewContext, viewCamera);
			glPopMatrix();

			glPushMatrix();
				doRenderStockScene(viewContext, viewCamera);
			glPopMatrix();

			doRenderScene(viewContext, viewCamera);
		glPopMatrix();
	}

	glFlush();

	// swap buffers
	viewContext.swapBuffer();

	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(oldMatrixMode);

#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			std::cerr << "OpenGL error at " << __LINE__ << " in " << __FILE__ << ": " << gluErrorString(glErrorCode) << std::endl;
	}
#endif
}

}  // namespace swl
