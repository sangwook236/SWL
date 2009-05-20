#include "swl/winview/WinViewCaptureApi.h"
#include "swl/winview/GdiBitmapBufferedContext.h"
#include "swl/winview/GdiplusBitmapBufferedContext.h"
#include "swl/winview/WinViewBase.h"
#include "swl/view/ViewCamera2.h"
#include "swl/utility/StringUtil.h"
#include <wingdi.h>
#include <gdiplus.h>

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif


namespace swl {

namespace {

// original codes are contained in IMA-GE toolkit
#if defined(_UNICODE) || defined(UNICODE)
bool writeRgbDib(const std::wstring &filePathName, const void *pixels, const long imgWidth, const long imgHeight, const int colorBitCount = 32)
#else
bool writeRgbDib(const std::string &filePathName, const void *pixels, const long imgWidth, const long imgHeight, const int colorBitCount = 32)
#endif
{
	if (filePathName.empty() || !pixels) return false;

    FILE *fp = NULL;
#if defined(_UNICODE) || defined(UNICODE)
	_wfopen_s(&fp, filePathName.c_str(), L"wb");
#else
	fopen_s(&fp, filePathName.c_str(), "wb");
#endif
	if (!fp) return false;

	// device information
    BITMAPINFOHEADER bmihDIB;
	memset(&bmihDIB, 0, sizeof(BITMAPINFOHEADER));

	// following routine aligns given value to 4 bytes boundary.
	// the current implementation of DIB rendering in Windows 95/98/NT seems to be free from this alignment
	// but old version compatibility is needed.
	const long width = ((imgWidth + 3) / 4 * 4 > 0) ? imgWidth : 4;
	const long height = 0 == imgHeight ? 1 : imgHeight;

	bmihDIB.biSize			= sizeof(BITMAPINFOHEADER);
	bmihDIB.biWidth			= width;
	bmihDIB.biHeight		= height;
	bmihDIB.biPlanes		= 1;
	//--S 2002/05/07: Sang-Wook Lee
	//bmihDIB.biBitCount	= 24;
	bmihDIB.biBitCount		= colorBitCount; //32;
	//--E 2002/05/07
	bmihDIB.biCompression	= BI_RGB;
	bmihDIB.biSizeImage		= 0;  // for BI_RGB
	//bmihDIB.biSizeImage	= width * height * 3;

	//int bitspixel = GetDeviceCaps(BITSPIXEL);
	const int colorByteCount = colorBitCount / 8;

	// to be free from structure byte alignment,
	// member-to-member I/O transport is performed
	//--S 2002/05/07: Sang-Wook Lee
	//const DWORD imgSize = width * height * 3;  // file size [byte]
	const DWORD imgSize = width * height * colorByteCount;  // file size [byte]
	//--E 2002/05/07

    if (fwrite("BM", 2, 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}

    DWORD dwBuffer = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    if (fwrite(&dwBuffer, 4, 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}

    // reserved
    dwBuffer = 0;
    if (fwrite(&dwBuffer, 4, 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}

    // pixel offset
    dwBuffer = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    if (fwrite(&dwBuffer, 4, 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}

    // BITMAPINFOHEADER is dword aligned
	// so above process is not needed for the current little endian based implementation

    if (fwrite(&bmihDIB, sizeof(bmihDIB), 1, fp) != 1)
	{
		fclose(fp);
		return false;
	}

    // write pixels
    if(fwrite(pixels, 1, imgSize, fp) != imgSize)
	{
		fclose(fp);
		return false;
	}

    fclose(fp);
	return true;
}

int GetEncoderClsid(const wchar_t *format, CLSID *pClsid)
{
	UINT num = 0;  // number of image encoders
	UINT size = 0;  // size of the image encoder array in bytes

	Gdiplus::GetImageEncodersSize(&num, &size);
	if (0 == size)
		return -1;  // failure

	Gdiplus::ImageCodecInfo *pImageCodecInfo = (Gdiplus::ImageCodecInfo *)new char [size];
	if (NULL == pImageCodecInfo)
		return -1;  // failure

	Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);

	for (UINT j = 0; j < num; ++j)
	{
		if (_wcsicmp(pImageCodecInfo[j].MimeType, format) == 0)
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			delete [] (char *)pImageCodecInfo;
			return j;  // success
		}    
	}

	delete [] (char *)pImageCodecInfo;
	return -1;  // failure
}

}  // unnamed namespace

#if defined(_UNICODE) || defined(UNICODE)
bool captureWinViewUsingGdi(const std::wstring &filePathName, WinViewBase &view, HWND hWnd)
#else
bool captureWinViewUsingGdi(const std::string &filePathName, WinViewBase &view, HWND hWnd)
#endif
{
	if (filePathName.empty()) return false;

	const boost::shared_ptr<WinViewBase::context_type> &currContext = view.topContext();
	const boost::shared_ptr<WinViewBase::camera_type> &currCamera = view.topCamera();
 	if (!currContext || !currCamera) return false;

	const Region2<int> &viewport = currCamera->getViewport();
/*
	bool isCaptured = true;
	if (currContext->isOffScreenUsed())
	{
		try
		{
			const HDC *dc = boost::any_cast<const HDC *>(currContext->getNativeContext());
			if (dc)
			{
				const int colorBitCount = GetDeviceCaps(*dc, BITSPIXEL);
				// write DIB
				return writeRgbDib(filePathName, currContext->getOffScreen(), viewport.getWidth(), viewport.getHeight(), colorBitCount);
			}
		}
		catch (const boost::bad_any_cast &)
		{
			isCaptured = false;
		}
	}

	if (!isCaptured)
	{
*/
		GdiBitmapBufferedContext captureContext(hWnd, viewport, false);

		{
			ViewContextGuard guard(captureContext);
			view.initializeView();
			currCamera->setViewport(0, 0, viewport.getWidth(), viewport.getHeight());
			view.renderScene(captureContext, *currCamera);
		}

		// write DIB
		return writeRgbDib(filePathName, captureContext.getOffScreen(), viewport.getWidth(), viewport.getHeight());
//	}
	
	return true;
}

#if defined(_UNICODE) || defined(UNICODE)
bool captureWinViewUsingGdiplus(const std::wstring &filePathName, const std::wstring &fileExtName, WinViewBase &view, HWND hWnd)
#else
bool captureWinViewUsingGdiplus(const std::string &filePathName, const std::string &fileExtName, WinViewBase &view, HWND hWnd)
#endif
{
	if (filePathName.empty()) return false;

	const boost::shared_ptr<WinViewBase::context_type> &currContext = view.topContext();
	const boost::shared_ptr<WinViewBase::camera_type> &currCamera = view.topCamera();
 	if (!currContext || !currCamera) return false;

	const Region2<int> &viewport = currCamera->getViewport();

	//--S [] 2009/05/14: Sang-Wook Lee
/*
	bool isCaptured = true;
	if (currContext->isOffScreenUsed())
	{
		try
		{
			const HDC *dc = boost::any_cast<const HDC *>(currContext->getNativeContext());
			if (dc)
			{
				const int colorBitCount = GetDeviceCaps(*dc, BITSPIXEL);
				// write DIB
				return writeRgbDib(filePathName, currContext->getOffScreen(), viewport.getWidth(), viewport.getHeight(), colorBitCount);
			}
		}
		catch (const boost::bad_any_cast &)
		{
			isCaptured = false;
		}
	}

	if (!isCaptured)
	{
*/
	//--E [] 2009/05/14
		GdiplusBitmapBufferedContext captureContext(hWnd, viewport, false);

		{
			ViewContextGuard guard(captureContext);
			view.initializeView();
			currCamera->setViewport(0, 0, viewport.getWidth(), viewport.getHeight());
			view.renderScene(captureContext, *currCamera);
		}

		// write DIB
		if (captureContext.getOffScreen())
		{
			CLSID clsId;
			int ret = -1;
#if defined(_UNICODE) || defined(UNICODE)
			if (_wcsicmp(fileExtName.c_str(), L"bmp") == 0)
#else
			if (stricmp(fileExtName.c_str(), "bmp") == 0)
#endif
				ret = GetEncoderClsid(L"image/bmp", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"jpg") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "jpg") == 0)
#endif
				ret = GetEncoderClsid(L"image/jpeg", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"gif") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "gif") == 0)
#endif
				ret = GetEncoderClsid(L"image/gif", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"png") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "png") == 0)
#endif
				ret = GetEncoderClsid(L"image/png", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"tif") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "tif") == 0)
#endif
				ret = GetEncoderClsid(L"image/tiff", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"emf") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "emf") == 0)
#endif
				ret = GetEncoderClsid(L"image/x-emf", &clsId);
#if defined(_UNICODE) || defined(UNICODE)
			else if (_wcsicmp(fileExtName.c_str(), L"wmf") == 0)
#else
			else if (stricmp(fileExtName.c_str(), "wmf") == 0)
#endif
				ret = GetEncoderClsid(L"image/x-wmf", &clsId);

			if (-1 != ret)
#if defined(_UNICODE) || defined(UNICODE)
				return Gdiplus::Ok == captureContext.getOffScreen()->Save(filePathName.c_str(), &clsId, NULL);
#else
				return Gdiplus::Ok == captureContext.getOffScreen()->Save(StringUtil::mbs2wcs(filePathName).c_str(), &clsId, NULL);
#endif
			else return false;
		}
		else return false;
	//--S [] 2009/05/14: Sang-Wook Lee
/*
	}

	return true;
*/
	//--E [] 2009/05/14
}

}  // namespace swl
