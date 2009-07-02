#if !defined(__SWL__RESOURCE_LEAKAGE_CHECK__H_)
#define __SWL__RESOURCE_LEAKAGE_CHECK__H_ 1


#if (_MSC_VER >= 1500)  // VC2008
#if defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define DEBUG_NEW new(__FILE__, __LINE__)
#pragma comment(lib, "mfc90ud.lib")
#endif
#elif (_MSC_VER >= 1400)  // VC2005
#if defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define DEBUG_NEW new(__FILE__, __LINE__)
#pragma comment(lib, "mfc80ud.lib")
#endif
#elif (_MSC_VER >= 1310)  // VC2003
#if defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define DEBUG_NEW new(__FILE__, __LINE__)
#pragma comment(lib, "mfc71ud.lib")
#endif
#elif (_MSC_VER >= 1300)  // VC2002
#if defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define DEBUG_NEW new(__FILE__, __LINE__)
#pragma comment(lib, "mfc70ud.lib")
#endif
#elif (_MSC_VER >= 1200)  // VC6/eVC4
#if defined(_DEBUG)
#error VC6/eVC4 is not set properly yet.
#endif
#elif (_MSC_VER >= 1100)  // VC5
#if defined(_DEBUG)
#error VC5 is not set properly yet.
#endif
#elif (_MSC_VER >= 1000)  // VC4
#if defined(_DEBUG)
#error VC4 is not set properly yet.
#endif
#else
#error this compiler is not supported
#endif

#if (_MSC_VER < 1100)  // VC4 and before
#elif (_MSC_VER < 1200)  // VC5
#elif (_MSC_VER < 1300)  // VC6/eVC4 
#elif (_MSC_VER < 1310)  // VC2002
#elif (_MSC_VER < 1400)  // VC2003
#elif (_MSC_VER < 1500)  // VC2005
#elif (_MSC_VER >= 1500)  // VC2008 and higher
#endif


#endif  // __SWL__RESOURCE_LEAKAGE_CHECK__H_
