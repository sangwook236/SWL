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
	// if maxDisplayListCount_ == 0, OpenGL display list won't be used
	// if maxFontDisplayListCount_ == 0, OpenGL display list for fonts won't be used
	if (maxDisplayListCount_ < 0 || maxFontDisplayListCount_ < 0) return false;

	unsigned int displayListNameBase = 0, fontDisplayListNameBase = 0;

	if (isContextActivated)
	{
		if (maxDisplayListCount_ > 0) displayListNameBase = glGenLists(maxDisplayListCount_);
		if (maxFontDisplayListCount_ > 0) fontDisplayListNameBase = glGenLists(maxFontDisplayListCount_);
	}
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context.get()) return false;
		context_type::guard_type guard(*context);

		if (maxDisplayListCount_ > 0) displayListNameBase = glGenLists(maxDisplayListCount_);
		if (maxFontDisplayListCount_ > 0) fontDisplayListNameBase = glGenLists(maxFontDisplayListCount_);
	}

	// if displayListNameBase == 0, OpenGL display list won't be used.
	// if fontDisplayListNameBase == 0, OpenGL display list for fonts won't be used.
#if 0
	if (displayListNameBase)
		displayListStack_.push(displayListNameBase);
	if (fontDisplayListNameBase)
		fontDisplayListStack_.push(fontDisplayListNameBase);
	return 0 != displayListNameBase && 0 != fontDisplayListNameBase;
#else
	displayListStack_.push(displayListNameBase);
	fontDisplayListStack_.push(fontDisplayListNameBase);
	return (0 == maxDisplayListCount_ || 0 != displayListNameBase) &&
		(0 == maxFontDisplayListCount_ || 0 != fontDisplayListNameBase);
#endif
}

bool WglViewBase::popDisplayList(const bool isContextActivated)
{
	if (displayListStack_.empty() || fontDisplayListStack_.empty()) return false;

	const unsigned int currDisplayListNameBase = displayListStack_.top();
	const unsigned int currFontDisplayListNameBase = fontDisplayListStack_.top();

	// if currDisplayListNameBase != 0, OpenGL display list has been used.
	// if currFontDisplayListNameBase != 0, OpenGL display list for fonts has been used.
	if (isContextActivated)
	{
		if (currDisplayListNameBase)
			glDeleteLists(currDisplayListNameBase, maxDisplayListCount_);
		else displayListStack_.pop();
		if (currFontDisplayListNameBase)
			glDeleteLists(currFontDisplayListNameBase, maxFontDisplayListCount_);
		else fontDisplayListStack_.pop();
	}
	else
	{
		const boost::shared_ptr<context_type> &context = topContext();
		if (!context.get()) return false;
		context_type::guard_type guard(*context);

		if (currDisplayListNameBase)
			glDeleteLists(currDisplayListNameBase, maxDisplayListCount_);
		else displayListStack_.pop();
		if (currFontDisplayListNameBase)
			glDeleteLists(currFontDisplayListNameBase, maxFontDisplayListCount_);
		else fontDisplayListStack_.pop();
	}
	
	return true;
}

bool WglViewBase::isDisplayListUsed() const
{
	return (!displayListStack_.empty() && 0 != displayListStack_.top()) ||
		(!fontDisplayListStack_.empty() && 0 != fontDisplayListStack_.top());
}

unsigned int WglViewBase::getCurrentDisplayListNameBase() const
{
	return displayListStack_.empty() ? 0 : displayListStack_.top();
}

unsigned int WglViewBase::getCurrentFontDisplayListNameBase() const
{
	return fontDisplayListStack_.empty() ? 0 : fontDisplayListStack_.top();
}

}  // namespace swl
