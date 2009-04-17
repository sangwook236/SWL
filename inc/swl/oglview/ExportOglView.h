#if !defined(__SWL_OGL_VIEW__EXPORT_OGL_VIEW__H_)
#define __SWL_OGL_VIEW__EXPORT_OGL_VIEW__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_OGL_VIEW)
#		    define SWL_OGL_VIEW_API __declspec(dllexport)
#			define SWL_OGL_VIEW_EXPORT_TEMPLATE
#		else
#		    define SWL_OGL_VIEW_API __declspec(dllimport)
#			define SWL_OGL_VIEW_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_OGL_VIEW
#	else
#		define SWL_OGL_VIEW_API
#		define SWL_OGL_VIEW_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_OGL_VIEW)
#			define SWL_OGL_VIEW_API __declspec(dllexport)
#		else
#			define SWL_OGL_VIEW_API __declspec(dllimport)
#		endif  // EXPORT_SWL_OGL_VIEW
#	else
#		define SWL_OGL_VIEW_API
#	endif  // _USRDLL
#	define SWL_OGL_VIEW_EXPORT_TEMPLATE
#else
#   define SWL_OGL_VIEW_API
#	define SWL_OGL_VIEW_EXPORT_TEMPLATE
#endif


#endif  // __SWL_OGL_VIEW__EXPORT_OGL_VIEW__H_
