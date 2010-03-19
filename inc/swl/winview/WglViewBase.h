#if !defined(__SWL_WIN_VIEW__WGL_VIEW_BASE__H_)
#define __SWL_WIN_VIEW__WGL_VIEW_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewBase.h"


namespace swl {

class WglContextBase;
class GLCamera;

//-----------------------------------------------------------------------------------
// 

struct SWL_WIN_VIEW_API WglViewBase: public ViewBase<WglContextBase, GLCamera>
{
public:
	typedef ViewBase<context_type, camera_type> base_type;

public:
	WglViewBase(const int maxDisplayListCount, const int maxFontDisplayListCount)
	: maxDisplayListCount_(maxDisplayListCount), maxFontDisplayListCount_(maxFontDisplayListCount)
	{}
	virtual ~WglViewBase()  {}

public:
	void renderScene(context_type &context, camera_type &camera);

	virtual void pickObject(const int x, const int y, const bool isTemporary = false) = 0;
	virtual void pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary = false) = 0;

	//-------------------------------------------------------------------------
	// OpenGL display list

	bool pushDisplayList(const bool isContextActivated);
	bool popDisplayList(const bool isContextActivated);
	bool isDisplayListUsed() const;
	virtual bool createDisplayList(const bool isContextActivated) = 0;

	unsigned int getCurrentDisplayListNameBase() const;
	unsigned int getCurrentFontDisplayListNameBase() const;

private:
	virtual bool doPrepareRendering(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderStockScene(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderScene(const context_type &context, const camera_type &camera) = 0;

private:
	//-------------------------------------------------------------------------
	// OpenGL display list

	/// a stack of the name base of OpenGL display list. if the name base of OpenGL display list == 0, OpenGL display list isn't used.
	std::stack<unsigned int> displayListStack_;
	const int maxDisplayListCount_;
	/// a stack of the name base of OpenGL display list for fonts. if the name base of OpenGL display list for fonts == 0, OpenGL display list for fonts isn't used.
	std::stack<unsigned int> fontDisplayListStack_;
	const int maxFontDisplayListCount_;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_BASE__H_
