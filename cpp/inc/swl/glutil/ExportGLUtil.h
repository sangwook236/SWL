#if !defined(__SWL_GL_UTIL__EXPORT_GL_UTIL__H_)
#define __SWL_GL_UTIL__EXPORT_GL_UTIL__H_ 1


#if defined(_WIN32) || defined(WIN32) || defined(_WIN64) || defined(WIN64)
#	if defined(_MSC_VER)
#		if defined(SWL_GL_UTIL_EXPORT)
#		    define SWL_GL_UTIL_API __declspec(dllexport)
#			define SWL_GL_UTIL_TEMPLATE_EXTERN
#		else
#		    define SWL_GL_UTIL_API __declspec(dllimport)
#			define SWL_GL_UTIL_TEMPLATE_EXTERN extern
#		endif  // SWL_GL_UTIL_EXPORT
#	else
#		define SWL_GL_UTIL_API
#		define SWL_GL_UTIL_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_GL_UTIL_EXPORT)
#			define SWL_GL_UTIL_API __declspec(dllexport)
#		else
#			define SWL_GL_UTIL_API __declspec(dllimport)
#		endif  // SWL_GL_UTIL_EXPORT
#	else
#		define SWL_GL_UTIL_API
#	endif  // _USRDLL
#	define SWL_GL_UTIL_TEMPLATE_EXTERN
#else
#   define SWL_GL_UTIL_API
#	define SWL_GL_UTIL_TEMPLATE_EXTERN
#endif


#endif  // __SWL_GL_UTIL__EXPORT_GL_UTIL__H_
