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

public:
	void renderScene(context_type &context, camera_type &camera);

	virtual void pickObject(const int x, const int y, const bool isTemporary = false) = 0;
	virtual void pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary = false) = 0;

	virtual void dragObject(const int x1, const int y1, const int x2, const int y2) = 0;

private:
	virtual bool doPrepareRendering(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderStockScene(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderScene(const context_type &context, const camera_type &camera) = 0;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WIN_VIEW_BASE__H_
