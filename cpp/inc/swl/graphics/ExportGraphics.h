#if !defined(__SWL_GRAPHICS__EXPORT_GRAPHICS__H_)
#define __SWL_GRAPHICS__EXPORT_GRAPHICS__H_ 1


#if defined(WIN32) || defined(_WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_GRAPHICS_EXPORT)
#		    define SWL_GRAPHICS_API __declspec(dllexport)
#			define SWL_GRAPHICS_TEMPLATE_EXTERN
#		else
#		    define SWL_GRAPHICS_API __declspec(dllimport)
#			define SWL_GRAPHICS_TEMPLATE_EXTERN extern
#		endif  // SWL_GRAPHICS_EXPORT
#	else
#		define SWL_GRAPHICS_API
#		define SWL_GRAPHICS_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_GRAPHICS_EXPORT)
#			define SWL_GRAPHICS_API __declspec(dllexport)
#		else
#			define SWL_GRAPHICS_API __declspec(dllimport)
#		endif  // SWL_GRAPHICS_EXPORT
#	else
#		define SWL_GRAPHICS_API
#	endif  // _USRDLL
#	define SWL_GRAPHICS_TEMPLATE_EXTERN
#else
#   define SWL_GRAPHICS_API
#	define SWL_GRAPHICS_TEMPLATE_EXTERN
#endif


#endif  // __SWL_GRAPHICS__EXPORT_GRAPHICS__H_