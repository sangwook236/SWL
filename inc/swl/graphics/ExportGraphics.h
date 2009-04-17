#if !defined(__SWL_GRAPHICS__EXPORT_GRAPHICS__H_)
#define __SWL_GRAPHICS__EXPORT_GRAPHICS__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_GRAPHICS)
#		    define SWL_GRAPHICS_API __declspec(dllexport)
#			define SWL_GRAPHICS_EXPORT_TEMPLATE
#		else
#		    define SWL_GRAPHICS_API __declspec(dllimport)
#			define SWL_GRAPHICS_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_GRAPHICS
#	else
#		define SWL_GRAPHICS_API
#		define SWL_GRAPHICS_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_GRAPHICS)
#			define SWL_GRAPHICS_API __declspec(dllexport)
#		else
#			define SWL_GRAPHICS_API __declspec(dllimport)
#		endif  // EXPORT_SWL_GRAPHICS
#	else
#		define SWL_GRAPHICS_API
#	endif  // _USRDLL
#	define SWL_GRAPHICS_EXPORT_TEMPLATE
#else
#   define SWL_GRAPHICS_API
#	define SWL_GRAPHICS_EXPORT_TEMPLATE
#endif


#endif  // __SWL_GRAPHICS__EXPORT_GRAPHICS__H_