#if !defined(__SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_)
#define __SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_MACHINE_LEARNING)
#		    define SWL_MACHINE_LEARNING_API __declspec(dllexport)
#			define SWL_MACHINE_LEARNING_EXPORT_TEMPLATE
#		else
#		    define SWL_MACHINE_LEARNING_API __declspec(dllimport)
#			define SWL_MACHINE_LEARNING_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_MACHINE_LEARNING
#	else
#		define SWL_MACHINE_LEARNING_API
#		define SWL_MACHINE_LEARNING_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_MACHINE_LEARNING)
#			define SWL_MACHINE_LEARNING_API __declspec(dllexport)
#		else
#			define SWL_MACHINE_LEARNING_API __declspec(dllimport)
#		endif  // EXPORT_SWL_MACHINE_LEARNING
#	else
#		define SWL_MACHINE_LEARNING_API
#	endif  // _USRDLL
#	define SWL_MACHINE_LEARNING_EXPORT_TEMPLATE
#else
#   define SWL_MACHINE_LEARNING_API
#	define SWL_MACHINE_LEARNING_EXPORT_TEMPLATE
#endif


#endif  // __SWL_MACHINE_LEARNING__EXPORT_MACHINE_LEARNING__H_
