#if !defined(__SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_)
#define __SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_ 1


#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_MACHINE_VISION_EXPORT)
#		    define SWL_MACHINE_VISION_API __declspec(dllexport)
#			define SWL_MACHINE_VISION_TEMPLATE_EXTERN
#		else
#		    define SWL_MACHINE_VISION_API __declspec(dllimport)
#			define SWL_MACHINE_VISION_TEMPLATE_EXTERN extern
#		endif  // SWL_MACHINE_VISION_EXPORT
#	else
#		define SWL_MACHINE_VISION_API
#		define SWL_MACHINE_VISION_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_MACHINE_VISION_EXPORT)
#			define SWL_MACHINE_VISION_API __declspec(dllexport)
#		else
#			define SWL_MACHINE_VISION_API __declspec(dllimport)
#		endif  // SWL_MACHINE_VISION_EXPORT
#	else
#		define SWL_MACHINE_VISION_API
#	endif  // _USRDLL
#	define SWL_MACHINE_VISION_TEMPLATE_EXTERN
#else
#   define SWL_MACHINE_VISION_API
#	define SWL_MACHINE_VISION_TEMPLATE_EXTERN
#endif


#endif  // __SWL_MACHINE_VISION__EXPORT_MACHINE_VISION__H_
