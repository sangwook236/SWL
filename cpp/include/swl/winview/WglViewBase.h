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
	WglViewBase()  {}
	virtual ~WglViewBase()  {}

public:
	void renderScene(context_type &context, camera_type &camera);

	virtual bool createDisplayList(const bool isContextActivated) = 0;
	virtual void generateDisplayListName(const bool isContextActivated) = 0;
	virtual void deleteDisplayListName(const bool isContextActivated) = 0;
	virtual bool isDisplayListUsed() const = 0;

	virtual void pickObject(const int x, const int y, const bool isTemporary = false) = 0;
	virtual void pickObject(const int x1, const int y1, const int x2, const int y2, const bool isTemporary = false) = 0;

	virtual void dragObject(const int x1, const int y1, const int x2, const int y2) = 0;

private:
	virtual bool doPrepareRendering(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderStockScene(const context_type &context, const camera_type &camera) = 0;
	virtual bool doRenderScene(const context_type &context, const camera_type &camera) = 0;
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_VIEW_BASE__H_
