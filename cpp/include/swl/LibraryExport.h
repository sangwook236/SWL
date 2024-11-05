#if !defined(__SWL__LIBRARY_EXPORT__H_)
#define __SWL__LIBRARY_EXPORT__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_LIBRARY_EXPORT)
#		    define SWL_LIBRARY_API __declspec(dllexport)
#			define SWL_LIBRARY_TEMPLATE_EXTERN
#		else
#		    define SWL_LIBRARY_API __declspec(dllimport)
#			define SWL_LIBRARY_TEMPLATE_EXTERN extern
#		endif  // SWL_LIBRARY_EXPORT
#	else
#		define SWL_LIBRARY_API
#		define SWL_LIBRARY_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_LIBRARY_EXPORT)
#			define SWL_LIBRARY_API __declspec(dllexport)
#		else
#			define SWL_LIBRARY_API __declspec(dllimport)
#		endif  // SWL_LIBRARY_EXPORT
#	else
#		define SWL_LIBRARY_API
#	endif  // _USRDLL
#	define SWL_LIBRARY_TEMPLATE_EXTERN
#else
#   define SWL_LIBRARY_API
#	define SWL_LIBRARY_TEMPLATE_EXTERN
#endif


#endif  // __SWL__LIBRARY_EXPORT__H_
