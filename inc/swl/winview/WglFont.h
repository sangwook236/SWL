#if !defined(__SWL_WIN_VIEW__WGL_FONT__H_)
#define __SWL_WIN_VIEW__WGL_FONT__H_ 1


#include "swl/winview/ExportWinView.h"
#include "swl/glutil/GLDisplayListCallableInterface.h"
#include "swl/graphics/GraphicsObj.h"
#include <string>
#if defined(WIN32)
#include <windows.h>
#endif

namespace swl {

//-----------------------------------------------------------------------------------------
// class WglFont

class SWL_WIN_VIEW_API WglFont: public GraphicsObj, public GLDisplayListCallableInterface
{
public:
	typedef GraphicsObj base_type;

private:
	WglFont();
public:
	virtual ~WglFont();

private:
	WglFont(const WglFont &rhs);
	WglFont & operator=(const WglFont &rhs);

public:
	static WglFont & getInstance();
	static void clearInstance();

public:
	void setDeviceContext(HDC hDC)  {  hDC_ = hDC;  }

	/*virtual*/ void draw() const
	{
		throw std::runtime_error("this function should not be called");
	}

	/*virtual*/ bool createDisplayList();
	/*virtual*/ void callDisplayList() const
	{
		throw std::runtime_error("this function should not be called");
	}

	/*virtual*/ void processToPick(const int /*x*/, const int /*y*/, const int /*width*/, const int /*height*/) const
	{
		throw std::runtime_error("this function should not be called");
	}

#if defined(_UNICODE) || defined(UNICODE)
	void drawText(const float x, const float y, const float z, const std::wstring &str) const;
#else
	void drawText(const float x, const float y, const float z, const std::string &str) const;
#endif

private:
#if defined(_UNICODE) || defined(UNICODE)
	bool createWglBitmapFonts(HDC hDC, const std::wstring &fontName, const int fontSize) const;
	bool createWglOutlineFonts(HDC hDC, const std::wstring &fontName, const int fontSize, const float depth) const;

	void drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const;
	void drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const;
	void drawTextUsingWglBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const;
	void drawTextUsingWglOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const;
#else
	bool createWglBitmapFonts(HDC hDC, const std::string &fontName, const int fontSize) const;
	bool createWglOutlineFonts(HDC hDC, const std::string &fontName, const int fontSize, const float depth) const;

	void drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::string &str) const;
	void drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const;
	void drawTextUsingWglBitmapFonts(const float x, const float y, const float z, const std::string &str) const;
	void drawTextUsingWglOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const;
#endif

private:
	static WglFont *singleton_;

	HDC hDC_;

	// for WGL bitmap fonts
	static const int MAX_WGL_BITMAP_FONT_DISPLAY_LIST_COUNT = 96;
	// for WGL outline fonts
	static const int MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT = 256;
	mutable GLYPHMETRICSFLOAT gmf_[MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT];
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_FONT__H_
