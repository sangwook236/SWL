#if !defined(__SWL_KINEMATICS__EXPORT_KINEMATICS__H_)
#define __SWL_KINEMATICS__EXPORT_KINEMATICS__H_ 1


#if defined(WIN32) || defined(_WIN32)
#	if defined(_MSC_VER)
#		if defined(SWL_KINEMATICS_EXPORT)
#		    define SWL_KINEMATICS_API __declspec(dllexport)
#			define SWL_KINEMATICS_TEMPLATE_EXTERN
#		else
#		    define SWL_KINEMATICS_API __declspec(dllimport)
#			define SWL_KINEMATICS_TEMPLATE_EXTERN extern
#		endif  // SWL_KINEMATICS_EXPORT
#	else
#		define SWL_KINEMATICS_API
#		define SWL_KINEMATICS_TEMPLATE_EXTERN
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(SWL_KINEMATICS_EXPORT)
#			define SWL_KINEMATICS_API __declspec(dllexport)
#		else
#			define SWL_KINEMATICS_API __declspec(dllimport)
#		endif  // SWL_KINEMATICS_EXPORT
#	else
#		define SWL_KINEMATICS_API
#	endif  // _USRDLL
#	define SWL_KINEMATICS_TEMPLATE_EXTERN
#else
#   define SWL_KINEMATICS_API
#	define SWL_KINEMATICS_TEMPLATE_EXTERN
#endif


#endif  // __SWL_KINEMATICS__EXPORT_KINEMATICS__H_
