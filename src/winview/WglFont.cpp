#include "swl/Config.h"
#include "swl/winview/WglFont.h"
#include <GL/glut.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#define __USE_OPENGL_DISPLAY_LIST 1

//#define __USE_GLUT_BITMAP_FONTS 1
//#define __USE_GLUT_STROKE_FONTS 1
#define __USE_WGL_BITMAP_FONTS 1
//#define __USE_WGL_OUTLINE_FONTS 1

#if defined(__USE_GLUT_BITMAP_FONTS)
#define __USE_OPENGL_FONTS 1  // for GLUT bitmap fonts
#elif defined(__USE_GLUT_STROKE_FONTS)
#define __USE_OPENGL_FONTS 2  // for GLUT stroke fonts
#elif defined(__USE_WGL_BITMAP_FONTS) && defined(__USE_OPENGL_DISPLAY_LIST)
#define __USE_OPENGL_FONTS 3  // for WGL bitmap fonts
#elif defined(__USE_WGL_OUTLINE_FONTS) && defined(__USE_OPENGL_DISPLAY_LIST)
#define __USE_OPENGL_FONTS 4  // for WGL outline fonts
#endif


namespace swl {

//--------------------------------------------------------------------------
// class WglFont

/*static*/ WglFont *WglFont::singleton_ = NULL;


WglFont::WglFont()
: base_type(true, false),
  hDC_(NULL),
#if defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 3  // for WGL bitmap fonts
  GLDisplayListCallableInterface(MAX_WGL_BITMAP_FONT_DISPLAY_LIST_COUNT)
#elif defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 4  // for WGL outline fonts
  GLDisplayListCallableInterface(MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT)
#else
  GLDisplayListCallableInterface(0u)
#endif
{
}

WglFont::~WglFont()
{
}

/*static*/ WglFont & WglFont::getInstance()
{
	if (NULL == singleton_)
		singleton_ = new WglFont();

	return *singleton_;
}

/*static*/ void WglFont::clearInstance()
{
	if (singleton_)
	{
		delete singleton_;
		singleton_ = NULL;
	}
}

bool WglFont::createDisplayList()
{
	if (NULL == hDC_) return false;

#if defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 3  // for WGL bitmap fonts
#if defined(_UNICODE) || defined(UNICODE)
	return createWglBitmapFonts(hDC_, L"Comic Sans MS", 24);
#else
	return createWglBitmapFonts(hDC_, "Comic Sans MS", 24);
#endif
#elif defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 4  // for WGL outline fonts
#if defined(_UNICODE) || defined(UNICODE)
	return createWglOutlineFonts(hDC_, L"Arial", 10, 0.25f);
#else
	return createWglOutlineFonts(hDC_, "Arial", 10, 0.25f);
#endif
#else
	return true;
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
bool WglFont::createWglBitmapFonts(HDC hDC, const std::wstring &fontName, const int fontSize) const
#else
bool WglFont::createWglBitmapFonts(HDC hDC, const std::string &fontName, const int fontSize) const
#endif
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
#if defined(_UNICODE) || defined(UNICODE)
	const DWORD charSet = 0 == _wcsicmp(fontName.c_str(), L"symbol") ? SYMBOL_CHARSET : ANSI_CHARSET;
#else
	const DWORD charSet = 0 == stricmp(fontName.c_str(), "symbol") ? SYMBOL_CHARSET : ANSI_CHARSET;
#endif
	const HFONT hFont = CreateFont(
		fontSize, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
		charSet, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
		ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH,
		fontName.c_str()
	);
	if (!hFont) return false;
	const HFONT hOldFont = (HFONT)SelectObject(hDC, hFont);

	const bool ret = TRUE == wglUseFontBitmaps(hDC, 32, MAX_WGL_BITMAP_FONT_DISPLAY_LIST_COUNT, getDisplayListNameBase());

	SelectObject(hDC, hOldFont);
	return ret;
#else
	return false;
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
bool WglFont::createWglOutlineFonts(HDC hDC, const std::wstring &fontName, const int fontSize, const float depth) const
#else
bool WglFont::createWglOutlineFonts(HDC hDC, const std::string &fontName, const int fontSize, const float depth) const
#endif
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
#if defined(_UNICODE) || defined(UNICODE)
	const DWORD charSet = 0 == _wcsicmp(fontName.c_str(), L"symbol") ? SYMBOL_CHARSET : ANSI_CHARSET;
#else
	const DWORD charSet = 0 == stricmp(fontName.c_str(), "symbol") ? SYMBOL_CHARSET : ANSI_CHARSET;
#endif
	const HFONT hFont = CreateFont(
		fontSize, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE,
		charSet, OUT_TT_PRECIS, CLIP_DEFAULT_PRECIS,
		ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH,
		fontName.c_str()
	);
	if (!hFont) return false;
	const HFONT hOldFont = (HFONT)SelectObject(hDC, hFont);

	// it takes long time to create display lists for outline fonts
	const bool ret = TRUE == wglUseFontOutlines(hDC, 0, MAX_WGL_OUTLINE_FONT_DISPLAY_LIST_COUNT, getDisplayListNameBase(), 0.0f, depth, WGL_FONT_POLYGONS, gmf_);

	SelectObject(hDC, hOldFont);
	return ret;
#else
	return false;
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
void WglFont::drawText(const float x, const float y, const float z, const std::wstring &str) const
#else
void WglFont::drawText(const float x, const float y, const float z, const std::string &str) const
#endif
{
#if defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 1  // for GLUT bitmap fonts
	drawTextUsingGlutBitmapFonts(x, y, z, str);
#elif defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 2  // for GLUT stroke fonts
	const float scale = 2.0f;
	drawTextUsingGlutStrokeFonts(x, y, z, scale, scale, scale, str);
#elif defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 3  // for WGL bitmap fonts
	drawTextUsingWglBitmapFonts(x, y, z, str);
#elif defined(__USE_OPENGL_FONTS) && __USE_OPENGL_FONTS == 4  // for WGL outline fonts
	const float scale = 300.0f;
	drawTextUsingWglOutlineFonts(x, y, z, scale, scale, 1.0f, str);
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
void WglFont::drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const
#else
void WglFont::drawTextUsingGlutBitmapFonts(const float x, const float y, const float z, const std::string &str) const
#endif
{
	void *font = GLUT_BITMAP_HELVETICA_18;

	glRasterPos3f(x, y, z);

#if defined(_UNICODE) || defined(UNICODE)
	for (std::wstring::const_iterator it = str.begin(); it != str.end(); ++it)
#else
	for (std::string::const_iterator it = str.begin(); it != str.end(); ++it)
#endif
		glutBitmapCharacter(font, *it);
}

#if defined(_UNICODE) || defined(UNICODE)
void WglFont::drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const
#else
void WglFont::drawTextUsingGlutStrokeFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const
#endif
{
	void *font = GLUT_STROKE_ROMAN;

	glPushMatrix();
		glTranslatef(x, y, z);
		glScalef(xScale, yScale, zScale);

#if defined(_UNICODE) || defined(UNICODE)
		for (std::wstring::const_iterator it = str.begin(); it != str.end(); ++it)
#else
		for (std::string::const_iterator it = str.begin(); it != str.end(); ++it)
#endif
			glutStrokeCharacter(font, *it);
	glPopMatrix();
}

#if defined(_UNICODE) || defined(UNICODE)
void WglFont::drawTextUsingWglBitmapFonts(const float x, const float y, const float z, const std::wstring &str) const
#else
void WglFont::drawTextUsingWglBitmapFonts(const float x, const float y, const float z, const std::string &str) const
#endif
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (isDisplayListUsed() && !str.empty())
	{
		glRasterPos3f(x, y, z);

		glPushAttrib(GL_LIST_BIT);
			glListBase(getDisplayListNameBase() - 32);
#if defined(_UNICODE) || defined(UNICODE)
			glCallLists((int)str.length(), GL_UNSIGNED_SHORT, str.c_str());
#else
			glCallLists((int)str.length(), GL_UNSIGNED_BYTE, str.c_str());
#endif
		glPopAttrib();
	}
#endif
}

#if defined(_UNICODE) || defined(UNICODE)
void WglFont::drawTextUsingWglOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::wstring &str) const
#else
void WglFont::drawTextUsingWglOutlineFonts(const float x, const float y, const float z, const float xScale, const float yScale, const float zScale, const std::string &str) const
#endif
{
#if defined(__USE_OPENGL_DISPLAY_LIST)
	if (isDisplayListUsed() && !str.empty())
	{
		float length = 0.0f;
		for (size_t i = 0; i < str.length(); ++i)
			length += gmf_[str[i]].gmfCellIncX;

		// save states
		GLint oldCullFace;
		glGetIntegerv(GL_CULL_FACE_MODE, &oldCullFace);
		if (GL_BACK != oldCullFace) glCullFace(GL_BACK); 
		const GLboolean isCullFace = glIsEnabled(GL_CULL_FACE);
		if (!isCullFace) glEnable(GL_CULL_FACE);

		glPushMatrix();
			glTranslatef(x, y, z);
			glScalef(xScale, yScale, zScale);

			glPushAttrib(GL_LIST_BIT);
				glListBase(getDisplayListNameBase());
#if defined(_UNICODE) || defined(UNICODE)
				glCallLists((int)str.length(), GL_UNSIGNED_SHORT, str.c_str());
#else
				glCallLists((int)str.length(), GL_UNSIGNED_BYTE, str.c_str());
#endif
			glPopAttrib();
		glPopMatrix();

		// restore states
		if (GL_BACK != oldCullFace) glCullFace(oldCullFace); 
		if (!isCullFace) glDisable(GL_CULL_FACE);
	}
#endif
}

}  // namespace swl
