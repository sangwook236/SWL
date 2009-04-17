#if !defined(__SWL_WIN_VIEW__EXPORT_WIN_VIEW_H_)
#define __SWL_WIN_VIEW__EXPORT_WIN_VIEW_H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_WIN_VIEW)
#		    define SWL_WIN_VIEW_API __declspec(dllexport)
#			define SWL_WIN_VIEW_EXPORT_TEMPLATE
#		else
#		    define SWL_WIN_VIEW_API __declspec(dllimport)
#			define SWL_WIN_VIEW_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_WIN_VIEW
#	else
#		define SWL_WIN_VIEW_API
#		define SWL_WIN_VIEW_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_WIN_VIEW)
#			define SWL_WIN_VIEW_API __declspec(dllexport)
#		else
#			define SWL_WIN_VIEW_API __declspec(dllimport)
#		endif  // EXPORT_SWL_WIN_VIEW
#	else
#		define SWL_WIN_VIEW_API
#	endif  // _USRDLL
#	define SWL_WIN_VIEW_EXPORT_TEMPLATE
#else
#   define SWL_WIN_VIEW_API
#	define SWL_WIN_VIEW_EXPORT_TEMPLATE
#endif


#endif  // __SWL_WIN_VIEW__EXPORT_WIN_VIEW_H_
