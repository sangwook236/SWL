#if !defined(__SWL_WIN_VIEW__WGL_FONT__H_)
#define __SWL_WIN_VIEW__WGL_FONT__H_ 1


#include "swl/winview/ExportWinView.h"
#include <string>
#include <windows.h>


namespace swl {

//-----------------------------------------------------------------------------------------
// class WglFont

class SWL_WIN_VIEW_API WglFont
{
public:
	//typedef WglFont base_type;

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
	bool create(HDC hDC, const unsigned int displayListNameBase);

	bool isDisplayListUsed() const  {  return 0 != displayListNameBase_;  }

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

public:
	static const int FONT_DISPLAY_LIST_COUNT;

private:
	static WglFont *singleton_;

	//
	unsigned int displayListNameBase_;

	// for WGL bitmap fonts
	static const int MAX_WGL_BITMAP_FONT_DISPLAY_LIST_COUNT = 96;
	// for WGL outline fonts
	static const int MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT = 256;
	mutable GLYPHMETRICSFLOAT gmf_[MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT];
};

}  // namespace swl


#endif  // __SWL_WIN_VIEW__WGL_FONT__H_
