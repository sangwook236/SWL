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
	WglViewBase()  {}
	virtual ~WglViewBase()  {}

protected:
	void renderScene(context_type &context, camera_type &camera);

private:
	virtual bool doPrepareRendering(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderStockScene(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderScene(const context_type &context, const camera_type &camera) = 0;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_BASE__H_
