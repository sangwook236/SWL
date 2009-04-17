#if !defined(__SWL_WIN_UTIL__WIN_REGISTRAR__H_ )
#define __SWL_WIN_UTIL__WIN_REGISTRAR__H_ 1


#include "swl/winutil/ExportWinUtil.h"
#include <windows.h>
#include <string>

namespace swl {

//-----------------------------------------------------------------------------------
//  WinRegistrar

class SWL_WIN_UTIL_API WinRegistrar  
{
public:
	enum ValueType { VT_ERROR = 0, VT_BINARY, VT_DWORD, VT_STRING, VT_STRING_LIST };

public:
#if defined(_UNICODE) || defined(UNICODE)
	typedef std::wstring subkey_type;
	typedef std::wstring string_type;
#else
	typedef std::string subkey_type;
	typedef std::string string_type;
#endif

public:
	WinRegistrar(const HKEY parentHkey, const subkey_type &key, bool isCreatedIfNew = true);
	explicit WinRegistrar(const WinRegistrar &rhs);
	~WinRegistrar();

private:
	WinRegistrar & operator=(const WinRegistrar &rhs);

public:
	bool close();

	DWORD getDisposition() const  {  return disposition_;  }

	//
	bool getSubkey(const int index, subkey_type &subkey) const;
	bool getFirstSubkey(subkey_type &subkey) const;
	bool getNextSubkey(subkey_type &subkey) const;
	bool deleteSubkey(const subkey_type &subkey) const;
	bool doesSubkeyExist(const subkey_type &subkey) const;

	//
	ValueType getValueType(const string_type &valueName) const;
	size_t getValueSize(const string_type &valueName) const;
	bool getValue(const int index, string_type &valueName) const;
	bool getFirstValue(subkey_type &valueName) const;
	bool getNextValue(string_type &valueName) const;
	bool deleteValue(const string_type &valueName) const;
	bool doesValueExist(const string_type &valueName) const;

	//
	bool readBinary(const subkey_type &subkey, unsigned char *value, size_t &len) const;
	bool writeBinary(const subkey_type &subkey, const unsigned char *value, const size_t len) const;
	DWORD readDWORD(const subkey_type &subkey, const DWORD defaultValue = -1) const;
	bool writeDWORD(const subkey_type &subkey, const DWORD value) const;
	string_type readString(const subkey_type &subkey, const string_type &defaultValue = subkey_type()) const;
	bool writeString(const subkey_type &subkey, const string_type &value) const;

private:
#if defined(_UNICODE) || defined(UNICODE)
	bool deleteSubkey(const HKEY hkey, const wchar_t *subkey) const;
#else
	bool deleteSubkey(const HKEY hkey, const char *subkey) const;
#endif

private:
	const subkey_type &key_;
	HKEY hkey_;
	DWORD disposition_;

	mutable int subkeyIndex_;
	mutable int valueIndex_;
};

}  // namespace swl


#endif  // __SWL_WIN_UTIL__WIN_REGISTRAR__H_ 
