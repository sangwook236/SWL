#if !defined(__SWL_WIN_VIEW__WGL_VIEW_BASE__H_)
#define __SWL_WIN_VIEW__WGL_VIEW_BASE__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/view/ViewBase.h"


namespace swl {

class WglContextBase;
class ViewCamera3;

//-----------------------------------------------------------------------------------
// 

struct SWL_WIN_VIEW_API WglViewBase: public ViewBase
{
public:
	typedef ViewBase base_type;

public:
	virtual ~WglViewBase()  {}

protected:
	void renderScene(swl::WglContextBase &context, swl::ViewCamera3 &camera);

private:
	virtual bool doPrepareRendering() = 0;
	virtual bool doRenderStockScene() = 0;
	virtual bool doRenderScene() = 0;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_BASE__H_
