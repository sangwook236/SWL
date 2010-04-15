#if !defined(__SWL_WIN_VIEW__GDI_RUBBER_BAND__H_ )
#define __SWL_WIN_VIEW__GDI_RUBBER_BAND__H_ 1


#include "swl/winview/GdiContextBase.h"


namespace swl {

//-----------------------------------------------------------------------------------
//

struct SWL_WIN_VIEW_API GdiRubberBand
{
	static void drawLine(HDC hdc, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase = true, const bool doesDraw = true);
	static void drawRectangle(HDC hdc, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase = true, const bool doesDraw = true);
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__GDI_RUBBER_BAND__H_
