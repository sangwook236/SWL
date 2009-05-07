#if !defined(__SWL_WIN_VIEW__WIN_VIEW_BASE__H_)
#define __SWL_WIN_VIEW__WIN_VIEW_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewBase.h"


namespace swl {

struct ViewContext;
class ViewCamera2;

//-----------------------------------------------------------------------------------
// 

struct SWL_WIN_VIEW_API WinViewBase: public ViewBase<ViewContext, ViewCamera2>
{
public:
	typedef ViewBase<context_type, camera_type> base_type;

public:
	WinViewBase()  {}
	virtual ~WinViewBase()  {}

protected:
	void renderScene();

private:
	virtual bool doPrepareRendering() = 0;
	virtual bool doRenderStockScene() = 0;
	virtual bool doRenderScene() = 0;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WIN_VIEW_BASE__H_
