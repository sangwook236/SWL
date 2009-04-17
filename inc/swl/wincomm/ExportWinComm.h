#if !defined(__SWL_WIN_COMM__EXPORT_SERIAL_COMM__H_)
#define __SWL_WIN_COMM__EXPORT_SERIAL_COMM__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_WIN_SERIAL_COMM)
#		    define SWL_WIN_COMM_API __declspec(dllexport)
#			define SWL_WIN_COMM_EXPORT_TEMPLATE
#		else
#		    define SWL_WIN_COMM_API __declspec(dllimport)
#			define SWL_WIN_COMM_EXPORT_TEMPLATE extern
#		endif  // EXPORT_WIN_SERIAL_COMM
#	else
#		define SWL_WIN_COMM_API
#		define SWL_WIN_COMM_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_WIN_SERIAL_COMM)
#			define SWL_WIN_COMM_API __declspec(dllexport)
#		else
#			define SWL_WIN_COMM_API __declspec(dllimport)
#		endif  // EXPORT_WIN_SERIAL_COMM
#	else
#		define SWL_WIN_COMM_API
#	endif  // _USRDLL
#	define SWL_WIN_COMM_EXPORT_TEMPLATE
#else
#   define SWL_WIN_COMM_API
#	define SWL_WIN_COMM_EXPORT_TEMPLATE
#endif


#endif  // __SWL_WIN_COMM__EXPORT_SERIAL_COMM__H_
