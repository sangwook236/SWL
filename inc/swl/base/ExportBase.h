#if !defined(__SWL_BASE__EXPORT_BASE__H_)
#define __SWL_BASE__EXPORT_BASE__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_BASE)
#		    define SWL_BASE_API __declspec(dllexport)
#			define SWL_BASE_EXPORT_TEMPLATE
#		else
#		    define SWL_BASE_API __declspec(dllimport)
#			define SWL_BASE_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_BASE
#	else
#		define SWL_BASE_API
#		define SWL_BASE_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_BASE)
#			define SWL_BASE_API __declspec(dllexport)
#		else
#			define SWL_BASE_API __declspec(dllimport)
#		endif  // EXPORT_SWL_BASE
#	else
#		define SWL_BASE_API
#	endif  // _USRDLL
#	define SWL_BASE_EXPORT_TEMPLATE
#else
#   define SWL_BASE_API
#	define SWL_BASE_EXPORT_TEMPLATE
#endif


#endif  // __SWL_BASE__EXPORT_BASE__H_
