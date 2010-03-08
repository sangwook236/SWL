#if !defined(__SWL_KINEMATICS__EXPORT_KINEMATICS__H_)
#define __SWL_KINEMATICS__EXPORT_KINEMATICS__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_KINEMATICS)
#		    define SWL_KINEMATICS_API __declspec(dllexport)
#			define SWL_KINEMATICS_EXPORT_TEMPLATE
#		else
#		    define SWL_KINEMATICS_API __declspec(dllimport)
#			define SWL_KINEMATICS_EXPORT_TEMPLATE extern
#		endif  // EXPORT_SWL_KINEMATICS
#	else
#		define SWL_KINEMATICS_API
#		define SWL_KINEMATICS_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_SWL_KINEMATICS)
#			define SWL_KINEMATICS_API __declspec(dllexport)
#		else
#			define SWL_KINEMATICS_API __declspec(dllimport)
#		endif  // EXPORT_SWL_KINEMATICS
#	else
#		define SWL_KINEMATICS_API
#	endif  // _USRDLL
#	define SWL_KINEMATICS_EXPORT_TEMPLATE
#else
#   define SWL_KINEMATICS_API
#	define SWL_KINEMATICS_EXPORT_TEMPLATE
#endif


#endif  // __SWL_KINEMATICS__EXPORT_KINEMATICS__H_
