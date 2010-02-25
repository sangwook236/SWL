#if !defined(__SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_)
#define __SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_MACHINE_VISION)
#		    define SWL_MACHINE_VISION_API __declspec(dllexport)
#			define SWL_MACHINE_VISION_EXPORT_TEMPLATE
#		else
#		    define SWL_MACHINE_VISION_API __declspec(dllimport)
#			define SWL_MACHINE_VISION_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_MACHINE_VISION
#	else
#		define SWL_MACHINE_VISION_API
#		define SWL_MACHINE_VISION_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_MACHINE_VISION)
#			define SWL_MACHINE_VISION_API __declspec(dllexport)
#		else
#			define SWL_MACHINE_VISION_API __declspec(dllimport)
#		endif  // EXPORT_SWL_MACHINE_VISION
#	else
#		define SWL_MACHINE_VISION_API
#	endif  // _USRDLL
#	define SWL_MACHINE_VISION_EXPORT_TEMPLATE
#else
#   define SWL_MACHINE_VISION_API
#	define SWL_MACHINE_VISION_EXPORT_TEMPLATE
#endif


#endif  // __SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_
