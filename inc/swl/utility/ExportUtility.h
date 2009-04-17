#if !defined(__SWL_UTILITY__EXPORT_UTILITY__H_)
#define __SWL_UTILITY__EXPORT_UTILITY__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_UTILITY)
#		    define SWL_UTILITY_API __declspec(dllexport)
#			define SWL_UTILITY_EXPORT_TEMPLATE
#		else
#		    define SWL_UTILITY_API __declspec(dllimport)
#			define SWL_UTILITY_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_UTILITY
#	else
#		define SWL_UTILITY_API
#		define SWL_UTILITY_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_UTILITY)
#			define SWL_UTILITY_API __declspec(dllexport)
#		else
#			define SWL_UTILITY_API __declspec(dllimport)
#		endif  // EXPORT_SWL_UTILITY
#	else
#		define SWL_UTILITY_API
#	endif  // _USRDLL
#	define SWL_UTILITY_EXPORT_TEMPLATE
#else
#   define SWL_UTILITY_API
#	define SWL_UTILITY_EXPORT_TEMPLATE
#endif


#endif  // __SWL_UTILITY__EXPORT_UTILITY__H_
