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

WglContextBase::WglContextBase(const Region2<int>& drawRegion, const bool isOffScreenUsed)
: base_type(drawRegion, isOffScreenUsed),
  wglRC_(NULL)
{
}

WglContextBase::WglContextBase(const RECT& drawRect, const bool isOffScreenUsed)
: base_type(Region2<int>(drawRect.left, drawRect.top, drawRect.right, drawRect.bottom), isOffScreenUsed),
  wglRC_(NULL)
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

/*static*/ void WglContextBase::usePalette(const HDC hDC, const PIXELFORMATDESCRIPTOR &pfd)
{
	// when using 256 color
    int nColorBit = GetDeviceCaps(hDC, BITSPIXEL);
    if (nColorBit <= 8 && !shPalette_)
	{
        // following routines are originated from ogl2 sdk made by Sillicon Graphics and modified for glext
        int nPalette = 1 << nColorBit;
        LOGPALETTE *pLogPalette = NULL;
		pLogPalette = (LOGPALETTE *)malloc(sizeof(LOGPALETTE) + nPalette * sizeof(PALETTEENTRY));
		
        if (pLogPalette)
		{
            pLogPalette->palVersion = 0x300;
            pLogPalette->palNumEntries = nPalette;
			
            // start with a copy of the current system palette examples/rb/rb.c
            // in ogl2 toolkit made by Sillicon Graphics
            GetSystemPaletteEntries(hDC, 0, nPalette, pLogPalette->palPalEntry);
			
            // fill in a rgba color palette
            const int rmask = (1 << pfd.cRedBits) - 1;
            const int gmask = (1 << pfd.cGreenBits) - 1;
            const int bmask = (1 << pfd.cBlueBits) - 1;
			
            for (int i = 0; i < nPalette; ++i)
			{
                pLogPalette->palPalEntry[i].peRed = (((i >> pfd.cRedShift) & rmask) * 255) / rmask;
                pLogPalette->palPalEntry[i].peGreen = (((i >> pfd.cGreenShift) & gmask) * 255) / gmask;
                pLogPalette->palPalEntry[i].peBlue = (((i >> pfd.cBlueShift) & bmask) * 255) / bmask;
                pLogPalette->palPalEntry[i].peFlags = 0;
            }
			
            shPalette_ = CreatePalette(pLogPalette);
            if (pLogPalette)
			{
				free(pLogPalette);
				pLogPalette = NULL;
			}
        }
    }
	
    if (shPalette_)
	{
        SelectPalette(hDC, shPalette_, FALSE);
        RealizePalette(hDC);
        ++sUsedPaletteCount_;
    }
}

}  // namespace swl
