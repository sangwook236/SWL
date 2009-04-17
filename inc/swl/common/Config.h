#if !defined(__SWL_COMMON__CONFIG__H_)
#define __SWL_COMMON__CONFIG__H_ 1


#if defined(_MSC_VER) && _MSC_VER == 1400
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#pragma comment(lib, "mfc80ud.lib")
#endif

namespace swl {

//--------------------------------------------------------------------------
// 


}  // namespace swl


#endif  // __SWL_COMMON__CONFIG__H_
