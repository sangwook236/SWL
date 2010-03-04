#if !defined(__SWL_WIN_VIEW__WGL_VIEW_BASE__H_)
#define __SWL_WIN_VIEW__WGL_VIEW_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewBase.h"


namespace swl {

class WglContextBase;
class OglCamera;

//-----------------------------------------------------------------------------------
// 

struct SWL_WIN_VIEW_API WglViewBase: public ViewBase<WglContextBase, OglCamera>
{
public:
	typedef ViewBase<context_type, camera_type> base_type;

public:
	WglViewBase(const int maxDisplayListCount)
	: maxDisplayListCount_(maxDisplayListCount)
	{}
	virtual ~WglViewBase()  {}

public:
	void renderScene(context_type &context, camera_type &camera);

	//-------------------------------------------------------------------------
	// OpenGL display list

	bool pushDisplayList(const bool isContextActivated, const bool disableDisplayList = false);
	bool popDisplayList(const bool isContextActivated);
	bool isDisplayListUsed() const;
	unsigned int getCurrentDisplayListNameBase() const;
	virtual bool createDisplayList(const bool isContextActivated) = 0;

private:
	virtual bool doPrepareRendering(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderStockScene(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderScene(const context_type &context, const camera_type &camera) = 0;

protected:
	//-------------------------------------------------------------------------
	// OpenGL display list

	/// a stack of the name base of OpenGL display list. if the name base of OpenGL display list == 0, OpenGL display list isn't used.
	std::stack<unsigned int> displayListStack_;
	const int maxDisplayListCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_BASE__H_
