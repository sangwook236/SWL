#if !defined(__SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_)
#define __SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_ 1


#if defined(WIN32) || defined(_WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_MACHINE_LEARNING_EXPORT)
#		    define SWL_MACHINE_LEARNING_API __declspec(dllexport)
#			define SWL_MACHINE_LEARNING_TEMPLATE_EXTERN
#		else
#		    define SWL_MACHINE_LEARNING_API __declspec(dllimport)
#			define SWL_MACHINE_LEARNING_TEMPLATE_EXTERN extern
#		endif  // SWL_MACHINE_LEARNING_EXPORT
#	else
#		define SWL_MACHINE_LEARNING_API
#		define SWL_MACHINE_LEARNING_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_MACHINE_LEARNING_EXPORT)
#			define SWL_MACHINE_LEARNING_API __declspec(dllexport)
#		else
#			define SWL_MACHINE_LEARNING_API __declspec(dllimport)
#		endif  // SWL_MACHINE_LEARNING_EXPORT
#	else
#		define SWL_MACHINE_LEARNING_API
#	endif  // _USRDLL
#	define SWL_MACHINE_LEARNING_TEMPLATE_EXTERN
#else
#   define SWL_MACHINE_LEARNING_API
#	define SWL_MACHINE_LEARNING_TEMPLATE_EXTERN
#endif


#endif  // __SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_
