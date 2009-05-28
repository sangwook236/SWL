#include "swl/winview/WglContextBase.h"
#include <iostream>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl  {

/*static*/ HGLRC WglContextBase::sSharedRC_ = NULL;
/*static*/ HPALETTE WglContextBase::shPalette_ = NULL;
/*static*/ size_t WglContextBase::sUsedPaletteCount_ = 0;

WglContextBase::WglContextBase(const Region2<int> &drawRegion, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(drawRegion, isOffScreenUsed, contextMode),
  wglRC_(NULL), isPaletteUsed_(false)
{
}

WglContextBase::WglContextBase(const RECT &drawRect, const bool isOffScreenUsed, const EContextMode contextMode)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), isOffScreenUsed, contextMode),
  wglRC_(NULL), isPaletteUsed_(false)
{
}

WglContextBase::~WglContextBase()
{
}

void WglContextBase::createDisplayList(const HDC hDC)
{
	// caution:
	//	OpenGL에서 display list를 share하고자 하는 경우 source RC와 destination RC가 동일하여야 한다.
	//	예를 들어, source RC의 flag 속성이 PFD_DRAW_TO_WINDOW이고 destination RC의 flag 속성이
	//	PFD_DRAW_TO_BITMAP이라면, display list의 share는 실패한다.

	// share display list
	if (!shareDisplayList(wglRC_))
	{
#if defined(WIN32) && defined(_DEBUG)
		LPVOID lpMsgBuf;
		FormatMessage( 
			FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			NULL,
			GetLastError(),
			MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),  //  Default language
			(LPTSTR)&lpMsgBuf,
			0,
			NULL 
		);
		// display the string
#if defined(_UNICODE) || defined(UNICODE)
		std::wcout << L"error : fail to share display lists(" << (LPCTSTR)lpMsgBuf << L") at " << __LINE__ << L" in " << __FILE__ << std::endl;
#else
		std::cerr << "error : fail to share display lists(" << (LPCTSTR)lpMsgBuf << ") at " << __LINE__ << " in " << __FILE__ << std::endl;
#endif
		// free the buffer
		LocalFree(lpMsgBuf);
#endif
		// need to add the code that the display lists are created
		if (wglMakeCurrent(hDC, wglRC_) == TRUE)
		{
			doRecreateDisplayList();
			wglMakeCurrent(NULL, NULL);
		}
	}
}

/*static*/ bool WglContextBase::shareDisplayList(HGLRC &wglRC)
{
	if (NULL == wglRC) return false;
	if (sSharedRC_ == wglRC) return true;
	if (NULL == sSharedRC_)
	{
		sSharedRC_ = wglRC;
		return true;
	}
	else return wglShareLists(sSharedRC_, wglRC) == TRUE;
}

/*static*/ void WglContextBase::createPalette(HDC hDC, const PIXELFORMATDESCRIPTOR &pfd, const int colorBitCount)
{
	// FIXME [check] >> this implementation is not tested
	if (colorBitCount <= 8 && !shPalette_)
	{
		// following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
		const int paletteSize = 1 << colorBitCount;
		LOGPALETTE *logPalette = (LOGPALETTE *)new char [sizeof(LOGPALETTE) + paletteSize * sizeof(PALETTEENTRY)];
		
		if (logPalette)
		{
			logPalette->palVersion = 0x300;
			// start with a copy of the current system palette examples/rb/rb.c
			// in ogl2 toolkit made by Sillicon Graphics
			logPalette->palNumEntries = GetSystemPaletteEntries(hDC, 0, paletteSize, logPalette->palPalEntry);

			// fill in a RGBA color palette
			const int rmask = (1 << pfd.cRedBits) - 1;
			const int gmask = (1 << pfd.cGreenBits) - 1;
			const int bmask = (1 << pfd.cBlueBits) - 1;
			
			for (int i = 0; i < paletteSize; ++i)
			{
				logPalette->palPalEntry[i].peRed = (((i >> pfd.cRedShift) & rmask) * 255) / rmask;
				logPalette->palPalEntry[i].peGreen = (((i >> pfd.cGreenShift) & gmask) * 255) / gmask;
				logPalette->palPalEntry[i].peBlue = (((i >> pfd.cBlueShift) & bmask) * 255) / bmask;
				logPalette->palPalEntry[i].peFlags = 0;
			}

			shPalette_ = CreatePalette(logPalette);

			delete [] (char *)logPalette;
			logPalette = NULL;
		}
	}

	if (shPalette_)
	{
		SelectPalette(hDC, shPalette_, FALSE);
		RealizePalette(hDC);
		++sUsedPaletteCount_;
	}
}

/*static*/ void WglContextBase::deletePalette(HDC hDC)
{
	// delete palette
	SelectPalette(hDC, (HPALETTE)GetStockObject(DEFAULT_PALETTE), FALSE);
	if (shPalette_ && --sUsedPaletteCount_ <= 0)
	{
		DeleteObject(shPalette_);
		shPalette_ = NULL;
		sUsedPaletteCount_ = 0;
	}
}

}  // namespace swl
