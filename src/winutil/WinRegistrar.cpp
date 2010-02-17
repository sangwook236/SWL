#include "swl/Config.h"
#include "swl/winutil/WinRegistrar.h"
#include "swl/base/LogException.h"
#include <winreg.h>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//  WinRegistrar

WinRegistrar::WinRegistrar(const HKEY parentHkey, const subkey_type &key, bool isCreatedIfNew /*= true*/)
: hkey_(NULL), disposition_(0), key_(key), subkeyIndex_(0), valueIndex_(0)
{
	if (isCreatedIfNew)
	{
		if (RegCreateKeyEx(parentHkey, key_.c_str(), 0, NULL, 0, KEY_ALL_ACCESS, NULL, &hkey_, &disposition_) != ERROR_SUCCESS)
		{
		}
	}
	else
	{
		if (RegOpenKeyEx(parentHkey, key_.c_str(), 0, KEY_ALL_ACCESS, &hkey_) != ERROR_SUCCESS)
		{
			hkey_ = NULL;
		}
	}
}

WinRegistrar::WinRegistrar(const WinRegistrar &rhs) 
: hkey_(NULL), disposition_(0), key_(rhs.key_), subkeyIndex_(0), valueIndex_(0)
{
	if (rhs.hkey_)
	{
		if (RegCreateKeyEx(rhs.hkey_, rhs.key_.c_str(), 0, NULL, 0, KEY_ALL_ACCESS, NULL, &hkey_, &disposition_) != ERROR_SUCCESS)
		{
		}
	}
	else
		hkey_ = NULL;
}

WinRegistrar::~WinRegistrar()
{
	close();
}

bool WinRegistrar::close()
{
	if (hkey_)
	{
		RegCloseKey(hkey_);
		hkey_ = 0;
		return true;
	}

	return false;
}


bool WinRegistrar::getSubkey(const int index, subkey_type &subkey) const
{
	DWORD len = 0;
#if defined(_UNICODE) || defined(UNICODE)
	wchar_t str[MAX_PATH] = { L'\0', };
#else
	char str[MAX_PATH] = { '\0', };
#endif

	const DWORD ret = RegEnumKeyEx(hkey_, index, str, &len, 0, NULL, NULL, NULL);
	if (ERROR_SUCCESS == ret)
	{
		subkey.assign(str, str + len);
		return true;
	}
	else
	{
		subkey.clear();
		if (ERROR_NO_MORE_ITEMS == ret)
			return false;  // no more subkeys
		else
			//throw std::runtime_error("can't find subkey");
			throw LogException(LogException::L_ERROR, "can't find subkey", __FILE__, __LINE__, __FUNCTION__);
	}
}

bool WinRegistrar::getFirstSubkey(subkey_type &subkey) const
{
	subkeyIndex_ = 0;

	return getSubkey(subkeyIndex_, subkey);
}

bool WinRegistrar::getNextSubkey(subkey_type &subkey) const
{
	++subkeyIndex_;

	return getSubkey(subkeyIndex_, subkey);
}

#if defined(_UNICODE) || defined(UNICODE)
bool WinRegistrar::deleteSubkey(const HKEY hkey, const wchar_t *subkey) const
#else
bool WinRegistrar::deleteSubkey(const HKEY hkey, const char *subkey) const
#endif
{
	if (!subkey)
		return false;
	else
	{
		// recursively delete any subkeys for the target subkey
		HKEY hSubkey = NULL;

		if (RegOpenKeyEx(hkey_, subkey, 0, KEY_ALL_ACCESS, &hSubkey) != ERROR_SUCCESS)
			return false;

		DWORD len = 0;
#if defined(_UNICODE) || defined(UNICODE)
		wchar_t str[MAX_PATH] = { L'\0', };
#else
		char str[MAX_PATH] = { '\0', };
#endif

		while (RegEnumKeyEx(hSubkey, 0, str, &len, 0, NULL, NULL, NULL) == ERROR_SUCCESS)
		{
			if (!deleteSubkey(hSubkey, str))
			{
				RegCloseKey(hSubkey);
				return false;
			}
		}

		RegCloseKey(hSubkey);
	}

	return RegDeleteKey(hkey_, subkey) == ERROR_SUCCESS;
}

bool WinRegistrar::deleteSubkey(const subkey_type &subkey) const
{
	return subkey.empty() ? false : deleteSubkey(hkey_, subkey.c_str());
}

bool WinRegistrar::doesSubkeyExist(const subkey_type &subkey) const
{
	if (subkey.empty()) return false;

	HKEY hSubkey = NULL;
	if (ERROR_SUCCESS != RegOpenKeyEx(hkey_, subkey.c_str(), 0, KEY_ALL_ACCESS, &hSubkey))
		return false;

	RegCloseKey(hSubkey);
	return true;
}

WinRegistrar::ValueType WinRegistrar::getValueType(const string_type &valueName) const
{
	if (valueName.empty()) return VT_ERROR;

	DWORD dwType = 0;
	if (RegQueryValueEx(hkey_, valueName.c_str(), 0, &dwType, NULL, NULL) != ERROR_SUCCESS)
		return VT_ERROR;

	switch (dwType)
	{
	case REG_BINARY:
		return VT_BINARY;
	case REG_DWORD:
	case REG_DWORD_BIG_ENDIAN:
		return VT_DWORD;
	case REG_EXPAND_SZ:
	case REG_SZ:
		return VT_STRING;
	case REG_MULTI_SZ:
		return VT_STRING_LIST;
	default:
		return VT_ERROR;  // there are other types, but not supported by WinRegistrar
	}
}

size_t WinRegistrar::getValueSize(const string_type &valueName) const
{
	if (valueName.empty()) return -1;

	DWORD dwSize = 0;
	RegQueryValueEx(hkey_, valueName.c_str(), 0, NULL, NULL, &dwSize);

	return (size_t)dwSize;
}

bool WinRegistrar::getValue(const int index, string_type &valueName) const
{
	DWORD len = 0;
#if defined(_UNICODE) || defined(UNICODE)
	wchar_t str[MAX_PATH] = { L'\0', };
#else
	char str[MAX_PATH] = { '\0', };
#endif

	if (RegEnumValue(hkey_, index, str, &len, 0, NULL, NULL, NULL) == ERROR_SUCCESS)
	{
		valueName.assign(str, str + len);
		return true;
	}
	else
	{
		valueName.clear();
		return false;  // no more subkeys
	}
}

bool WinRegistrar::getFirstValue(string_type &valueName) const
{
	valueIndex_ = 0;

	return getValue(valueIndex_, valueName);
}

bool WinRegistrar::getNextValue(string_type &valueName) const
{
	++valueIndex_;

	return getValue(valueIndex_, valueName);
}

bool WinRegistrar::deleteValue(const string_type &valueName) const
{
	return valueName.empty() ? false : RegDeleteValue(hkey_, valueName.c_str()) == ERROR_SUCCESS;
}

bool WinRegistrar::doesValueExist(const string_type &valueName) const
{
	if (valueName.empty()) return false;

	DWORD dwValueSize = 0;
	const DWORD dwType = REG_SZ;

	return RegQueryValueEx(hkey_, valueName.c_str(), 0, (LPDWORD)&dwType, NULL, &dwValueSize) == ERROR_SUCCESS;
}

bool WinRegistrar::readBinary(const subkey_type &subkey, unsigned char *value, size_t &len) const
{
	if (subkey.empty() || NULL == value) return false;

	const DWORD dwType = REG_BINARY;
	return RegQueryValueEx(hkey_, subkey.c_str(), 0, (LPDWORD)&dwType, (LPBYTE)value, (LPDWORD)&len) == ERROR_SUCCESS;
}

bool WinRegistrar::writeBinary(const subkey_type &subkey, const unsigned char *value, const size_t len) const
{
	if (subkey.empty() || NULL == value || len <= 0) return false;

	return RegSetValueEx(hkey_, subkey.c_str(), 0, REG_BINARY, (LPBYTE)value, (DWORD)len) == ERROR_SUCCESS;
}

DWORD WinRegistrar::readDWORD(const subkey_type &subkey, const DWORD defaultValue /*= -1*/) const
{
	if (subkey.empty()) return false;

	DWORD dwValue;
	DWORD dwValueSize = sizeof(DWORD);
	const DWORD dwType = REG_DWORD;

	if (RegQueryValueEx(hkey_, subkey.c_str(), 0, (LPDWORD)&dwType, (LPBYTE)&dwValue, &dwValueSize) == ERROR_SUCCESS)
		return dwValue;

	if (-1 != defaultValue)  // default specified
		RegSetValueEx(hkey_, subkey.c_str(), 0, REG_DWORD, (LPBYTE)&defaultValue, sizeof(DWORD));

	return defaultValue;
}

bool WinRegistrar::writeDWORD(const subkey_type &subkey, const DWORD value) const
{
	if (subkey.empty()) return false;

	return RegSetValueEx(hkey_, subkey.c_str(), 0, REG_DWORD, (LPBYTE)&value, sizeof(DWORD)) == ERROR_SUCCESS;
}

WinRegistrar::string_type WinRegistrar::readString(const subkey_type &subkey, const string_type &defaultValue /*= subkey_type()*/) const
{
	if (subkey.empty()) return false;

	DWORD dwValueSize = 0;
	DWORD dwType = REG_SZ;

	if (RegQueryValueEx(hkey_, subkey.c_str(), 0, &dwType, NULL, &dwValueSize) == ERROR_SUCCESS)
	{
		if (dwType != REG_SZ)
			return defaultValue.empty() ? string_type() : defaultValue;

#if defined(_UNICODE) || defined(UNICODE)
		dwValueSize *= sizeof(wchar_t);
#else
#endif
		unsigned char str[MAX_PATH] = { '\0', };

		if (RegQueryValueEx(hkey_, subkey.c_str(), 0, &dwType, (LPBYTE)str, &dwValueSize) == ERROR_SUCCESS)
		{
#if defined(_UNICODE) || defined(UNICODE)
			return string_type((wchar_t *)str);
#else
			return string_type((char *)str);
#endif
		}
		else
		{
			// do nothing
		}
	}

	// write the default value (if any)
	// if there was a value in the registry, it would already have been written
	if (defaultValue.empty())
		return string_type();
	else
	{
		return writeString(subkey, defaultValue) ? defaultValue : string_type();
	}
}

bool WinRegistrar::writeString(const subkey_type &subkey, const string_type &value) const
{
	if (subkey.empty() || value.empty()) return false;

#if defined(_UNICODE) || defined(UNICODE)
	return RegSetValueEx(hkey_, subkey.c_str(), 0, REG_SZ, (LPBYTE)value.c_str(), DWORD((value.length() + 1) * sizeof(wchar_t))) == ERROR_SUCCESS;
#else
	return RegSetValueEx(hkey_, subkey.c_str(), 0, REG_SZ, (LPBYTE)value.c_str(), DWORD((value.length() + 1) * sizeof(char))) == ERROR_SUCCESS;
#endif
}

}  // namespace swl
