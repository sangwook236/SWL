#if !defined(__SWL_MATH__EXPORT_MATH__H_)
#define __SWL_MATH__EXPORT_MATH__H_ 1


#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_MATH_EXPORT)
#		    define SWL_MATH_API __declspec(dllexport)
#			define SWL_MATH_TEMPLATE_EXTERN
#		else
#		    define SWL_MATH_API __declspec(dllimport)
#			define SWL_MATH_TEMPLATE_EXTERN extern
#		endif  // SWL_MATH_EXPORT
#	else
#		define SWL_MATH_API
#		define SWL_MATH_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_MATH_EXPORT)
#			define SWL_MATH_API __declspec(dllexport)
#		else
#			define SWL_MATH_API __declspec(dllimport)
#		endif  // SWL_MATH_EXPORT
#	else
#		define SWL_MATH_API
#	endif  // _USRDLL
#	define SWL_MATH_TEMPLATE_EXTERN
#else
#   define SWL_MATH_API
#	define SWL_MATH_TEMPLATE_EXTERN
#endif


#endif  // __SWL_MATH__EXPORT_MATH__H_