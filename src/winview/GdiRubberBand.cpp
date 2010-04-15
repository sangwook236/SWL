#include "swl/Config.h"
#include "swl/winview/GdiRubberBand.h"
#include <wingdi.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
// class GdiRubberBand

/*static*/ void GdiRubberBand::drawLine(HDC hdc, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase /*= true*/, const bool doesDraw /*= true*/)
{
	const int oldROP = GetROP2(hdc);
	SetROP2(hdc, R2_XORPEN);

	HPEN pen = CreatePen(PS_DOT, 1, RGB(255,255,255));
	const HPEN oldPen = (HPEN)SelectObject(hdc, pen);

	if (doesErase)
	{
		MoveToEx(hdc, initX, initY, NULL);
		LineTo(hdc, prevX, prevY);
	}

	if (doesDraw)
	{
		MoveToEx(hdc, initX, initY, NULL);
		LineTo(hdc, currX, currY);
	}

	SelectObject(hdc, oldPen);
	DeleteObject(pen);

	SetROP2(hdc, oldROP);
}

/*static*/ void GdiRubberBand::drawRectangle(HDC hdc, const int initX, const int initY, const int prevX, const int prevY, const int currX, const int currY, const bool doesErase /*= true*/, const bool doesDraw /*= true*/)
{
	if (doesErase)
	{
		const int left = prevX <= initX ? prevX : initX;
		const int right = prevX > initX ? prevX : initX;
		const int top = prevY <= initY ? prevY : initY;  // downward y-axis
		const int bottom = prevY > initY ? prevY : initY;  // downward y-axis

		RECT rect;
		rect.left = left;
		rect.right = right;
		rect.top = top;
		rect.bottom = bottom;
		DrawFocusRect(hdc, &rect);
	}

	if (doesDraw)
	{
		const int left = currX <= initX ? currX : initX;
		const int right = currX > initX ? currX : initX;
		const int top = currY <= initY ? currY : initY;  // downward y-axis
		const int bottom = currY > initY ? currY : initY;  // downward y-axis

		RECT rect;
		rect.left = left;
		rect.right = right;
		rect.top = top;
		rect.bottom = bottom;
		DrawFocusRect(hdc, &rect);
	}
}

}  // namespace swl
