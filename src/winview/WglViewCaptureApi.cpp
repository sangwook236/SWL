#include "swl/Config.h"
#include "swl/winview/WglViewCaptureApi.h"
#include "swl/winview/WglBitmapBufferedContext.h"
#include "swl/winview/WglViewBase.h"
#include "swl/oglview/OglCamera.h"
#include "swl/base/String.h"
#include <wingdi.h>
#include <gdiplus.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
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

}  // unnamed namespace

#if defined(_UNICODE) || defined(UNICODE)
bool captureWglViewUsingGdi(const std::wstring& filePathName, WglViewBase &view, HWND hWnd)
#else
bool captureWglViewUsingGdi(const std::string& filePathName, WglViewBase &view, HWND hWnd)
#endif
{
	if (filePathName.empty()) return false;

	const boost::shared_ptr<WglViewBase::context_type> &currContext = view.topContext();
	const boost::shared_ptr<WglViewBase::camera_type> &currCamera = view.topCamera();
 	if (!currContext || !currCamera) return false;

	const Region2<int> &viewport = currCamera->getViewport();
/*
	const boost::shared_ptr<WglViewBase::context_type> &currContext = view.topContext();
	const boost::shared_ptr<WglViewBase::camera_type> &currCamera = view.topCamera();
 	if (!currContext || !currCamera) return false;

	const Region2<int> &viewport = currCamera->getViewport();

	bool isCaptured = true;
	if (currContext->isOffScreenUsed())
	{
		try
		{
			const HDC *dc = boost::any_cast<HDC *>(currContext->getNativeContext());
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
		WglBitmapBufferedContext captureContext(hWnd, viewport, false);

		const bool isDisplayListShared = !currContext ? false : captureContext.shareDisplayList(*currContext);

		{
			WglBitmapBufferedContext::guard_type guard(captureContext);

			const bool doesRecreateDisplayListUsed = !isDisplayListShared && view.isDisplayListUsed();
			// create & push a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) view.pushDisplayList(true);

			view.initializeView();
			currCamera->setViewport(0, 0, viewport.getWidth(), viewport.getHeight());

			// re-create a OpenGL display list
			if (doesRecreateDisplayListUsed) view.createDisplayList(true);

			view.renderScene(captureContext, *currCamera);

			// pop & delete a new name base of OpenGL display list
			if (doesRecreateDisplayListUsed) view.popDisplayList(true);
		}

		// write DIB
		return writeRgbDib(filePathName, captureContext.getOffScreen(), viewport.getWidth(), viewport.getHeight());
//	}
	
	return true;
}

}  // namespace swl
