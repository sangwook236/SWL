#if !defined(__SWL_MATH__EXPORT_MATH__H_)
#define __SWL_MATH__EXPORT_MATH__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_MATH)
#		    define SWL_MATH_API __declspec(dllexport)
#			define SWL_MATH_EXPORT_TEMPLATE
#		else
#		    define SWL_MATH_API __declspec(dllimport)
#			define SWL_MATH_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_MATH
#	else
#		define SWL_MATH_API
#		define SWL_MATH_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_MATH)
#			define SWL_MATH_API __declspec(dllexport)
#		else
#			define SWL_MATH_API __declspec(dllimport)
#		endif  // EXPORT_SWL_MATH
#	else
#		define SWL_MATH_API
#	endif  // _USRDLL
#	define SWL_MATH_EXPORT_TEMPLATE
#else
#   define SWL_MATH_API
#	define SWL_MATH_EXPORT_TEMPLATE
#endif


#endif  // __SWL_MATH__EXPORT_MATH__H_