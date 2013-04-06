#include "swl/Config.h"
#include "swl/glutil/GLDisplayListCallableInterface.h"
#if defined(WIN32)
#include <windows.h>
#endif
#include <GL/gl.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GLDisplayListCallableInterface

GLDisplayListCallableInterface::GLDisplayListCallableInterface(const unsigned int displayListCount)
: //base_type(),
  displayListNameBaseStack_(), displayListCount_(displayListCount)
{
}

GLDisplayListCallableInterface::GLDisplayListCallableInterface(const GLDisplayListCallableInterface &rhs)
: //base_type(rhs),
  displayListNameBaseStack_(rhs.displayListNameBaseStack_), displayListCount_(rhs.displayListCount_)
{
}

GLDisplayListCallableInterface::~GLDisplayListCallableInterface()
{
}

bool GLDisplayListCallableInterface::pushDisplayList()
{
	// if displayListCount_ == 0u, OpenGL display list won't be used
	//if (displayListCount_ < 0u) return false;

	const unsigned int displayListNameBase = displayListCount_ > 0u ? glGenLists(displayListCount_) : 0u;

	// if displayListNameBase == 0u, OpenGL display list won't be used.
	displayListNameBaseStack_.push(displayListNameBase);
	return 0u == displayListCount_ || 0u != displayListNameBase;
}

bool GLDisplayListCallableInterface::popDisplayList()
{
	if (displayListNameBaseStack_.empty()) return false;

	const unsigned int currDisplayListNameBase = displayListNameBaseStack_.top();

	// if currDisplayListNameBase != 0u, OpenGL display list has been used.
	if (currDisplayListNameBase)
		glDeleteLists(currDisplayListNameBase, displayListCount_);
	displayListNameBaseStack_.pop();
	
	return true;
}

}  // namespace swl
