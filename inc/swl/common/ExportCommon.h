#if !defined(__SWL_COMMON__EXPORT_COMMON__H_)
#define __SWL_COMMON__EXPORT_COMMON__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_COMMON)
#		    define SWL_COMMON_API __declspec(dllexport)
#			define SWL_COMMON_EXPORT_TEMPLATE
#		else
#		    define SWL_COMMON_API __declspec(dllimport)
#			define SWL_COMMON_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_COMMON
#	else
#		define SWL_COMMON_API
#		define SWL_COMMON_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_COMMON)
#			define SWL_COMMON_API __declspec(dllexport)
#		else
#			define SWL_COMMON_API __declspec(dllimport)
#		endif  // EXPORT_SWL_COMMON
#	else
#		define SWL_COMMON_API
#	endif  // _USRDLL
#	define SWL_COMMON_EXPORT_TEMPLATE
#else
#   define SWL_COMMON_API
#	define SWL_COMMON_EXPORT_TEMPLATE
#endif


#endif  // __SWL_COMMON__EXPORT_COMMON__H_
