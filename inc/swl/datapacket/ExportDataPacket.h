#if !defined(__SWL_DATA_PACKET__EXPORT_DATA_PACKET__H_)
#define __SWL_DATA_PACKET__EXPORT_DATA_PACKET__H_ 1


#if defined(WIN32)
#	if defined(_MSC_VER)
#		if defined(EXPORT_SWL_DATA_PACKET)
#		    define SWL_DATA_PACKET_API __declspec(dllexport)
#			define SWL_DATA_PACKET_EXPORT_TEMPLATE
#		else
#		    define SWL_DATA_PACKET_API __declspec(dllimport)
#			define SWL_DATA_PACKET_EXPORT_TEMPLATE extern
#		endif  // EXPORT_PACKET_MODULE_MODULE
#	else
#		define SWL_DATA_PACKET_API
#		define SWL_DATA_PACKET_EXPORT_TEMPLATE
#	endif  // _MSC_VER
#elif defined(__MINGW32__)
#	if defined(_USRDLL)
#		if defined(EXPORT_PACKET_MODULE_MODULE)
#			define SWL_DATA_PACKET_API __declspec(dllexport)
#		else
#			define SWL_DATA_PACKET_API __declspec(dllimport)
#		endif  // EXPORT_PACKET_MODULE_MODULE
#	else
#		define SWL_DATA_PACKET_API
#	endif  // _USRDLL
#	define SWL_DATA_PACKET_EXPORT_TEMPLATE
#else
#   define SWL_DATA_PACKET_API
#	define SWL_DATA_PACKET_EXPORT_TEMPLATE
#endif


#endif  // __SWL_DATA_PACKET__EXPORT_DATA_PACKET__H_
