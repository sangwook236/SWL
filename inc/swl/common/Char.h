#if !defined(__SWL_COMMON__CHAR__H_)
#define __SWL_COMMON__CHAR__H_ 1

#include <string>

namespace swl {

//--------------------------------------------------------------------------
// 

#if defined(UNICODE) || defined(_UNICODE)
typedef wchar_t char_t
typedef std::wstring string_t
#else
typedef char char_t
typedef std::string string_t
#endif

}  // namespace swl


#endif  // __SWL_COMMON__CHAR__H_
