#if !defined(__SWL_BASE__EXPORT_BASE__H_)
#define __SWL_BASE__EXPORT_BASE__H_ 1


#if defined(WIN32) || defined(_WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_BASE_EXPORT)
#		    define SWL_BASE_API __declspec(dllexport)
#			define SWL_BASE_TEMPLATE_EXTERN
#		else
#		    define SWL_BASE_API __declspec(dllimport)
#			define SWL_BASE_TEMPLATE_EXTERN extern
#		endif  // SWL_BASE_EXPORT
#	else
#		define SWL_BASE_API
#		define SWL_BASE_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_BASE_EXPORT)
#			define SWL_BASE_API __declspec(dllexport)
#		else
#			define SWL_BASE_API __declspec(dllimport)
#		endif  // SWL_BASE_EXPORT
#	else
#		define SWL_BASE_API
#	endif  // _USRDLL
#	define SWL_BASE_TEMPLATE_EXTERN
#else
#   define SWL_BASE_API
#	define SWL_BASE_TEMPLATE_EXTERN
#endif


#endif  // __SWL_BASE__EXPORT_BASE__H_
