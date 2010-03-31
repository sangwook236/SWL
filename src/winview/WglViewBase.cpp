#include "swl/Config.h"
#include "swl/winview/WglViewBase.h"
#include "swl/winview/WglContextBase.h"
#include "swl/glutil/GLCamera.h"
#include <GL/glut.h>
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//  class WglViewBase

void WglViewBase::renderScene(context_type &context, camera_type &camera)
{
#ifdef _DEBUG
	{
		// error-checking routine of OpenGL
		const GLenum glErrorCode = glGetError();
		if (GL_NO_ERROR != glErrorCode)
			std::cerr << "OpenGL error at " << __LINE__ << " in " << __FILE__ << ": " << gluErrorString(glErrorCode) << std::endl;
	}
#endif

	GLint oldMatrixMode = 0;
	glGetIntegerv(GL_MATRIX_MODE, &oldMatrixMode);
	if (oldMatrixMode != GL_MODELVIEW) glMatrixMode(GL_MODELVIEW);

	{
		glPushMatrix();
			//
			glLoadIdentity();
			camera.lookAt();

			//
			glPushMatrix();
				doPrepareRendering(context, camera);
			glPopMatrix();

			glPushMatrix();
				doRenderStockScene(context, camera);
			glPopMatrix();

			doRenderScene(context, camera);
		glPopMatrix();
	}

	glFlush();

	// swap buffers
	context.swapBuffer();

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
