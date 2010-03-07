#include "swl/Config.h"
#include "swl/winview/WglViewBase.h"
#include "swl/winview/WglContextBase.h"
#include "swl/oglview/OglCamera.h"
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

	int oldMatrixMode = 0;
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

bool WglViewBase::pushDisplayList(const bool isContextActivated)
{
	unsigned int displayListNameBase = 0;

	if (isContextActivated)
		displayListNameBase = glGenLists(maxDisplayListCount_);
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context.get()) return false;

		context_type::guard_type guard(*context);
		displayListNameBase = glGenLists(maxDisplayListCount_);
	}

	// if displayListNameBase == 0, OpenGL display list won't be used.
#if 0
	if (displayListNameBase)
	{
		displayListStack_.push(displayListNameBase);
		return true;
	}
	else return false;
#else
	displayListStack_.push(displayListNameBase);
	return 0 == displayListNameBase;
#endif
}

bool WglViewBase::popDisplayList(const bool isContextActivated)
{
	if (displayListStack_.empty()) return false;

	const unsigned int currDisplayListNameBase = displayListStack_.top();

	// if currDisplayListNameBase == 0, OpenGL display list has been used.
	if (currDisplayListNameBase)
	{
		if (isContextActivated)
			glDeleteLists(currDisplayListNameBase, maxDisplayListCount_);
		else
		{
			const boost::shared_ptr<context_type> &context = topContext();
			if (!context.get()) return false;

			context_type::guard_type guard(*context);
			glDeleteLists(currDisplayListNameBase, maxDisplayListCount_);
		}

		displayListStack_.pop();
		return true;
	}
	else
	{
		displayListStack_.pop();
		return true;
	}
}

bool WglViewBase::isDisplayListUsed() const
{
	return !displayListStack_.empty() && 0 != displayListStack_.top();
}

unsigned int WglViewBase::getCurrentDisplayListNameBase() const
{
	return displayListStack_.empty() ? 0 : displayListStack_.top();
}

}  // namespace swl
