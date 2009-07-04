#include "swl/Config.h"
#include "swl/util/IniParser.h"
#include "swl/base/String.h"
#include "swl/base/LogException.h"
#include "iniparser_impl.h"


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

//--------------------------------------------------------------------------
//

#if defined(UNICODE) || defined(_UNICODE)
IniParser::IniParser(const std::wstring &iniFilePath)
#else
IniParser::IniParser(const std::string &iniFilePath)
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	ini_ = iniparser_load(String::wcs2mbs(iniFilePath).c_str());
#else
	ini_ = iniparser_load(iniFilePath.c_str());
#endif

	if (!ini_)
		throw LogException(LogException::L_ERROR, "ini file not opened", __FILE__, __LINE__, __FUNCTION__);
}

IniParser::~IniParser()
{
	iniparser_freedict(ini_);
}

int IniParser::getSectionCount() const
{
	return iniparser_getnsec(ini_);
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring IniParser::getSectionName(int n) const
#else
std::string IniParser::getSectionName(int n) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return String::mbs2wcs(iniparser_getsecname(ini_, n));
#else
	return std::string(iniparser_getsecname(ini_, n));
#endif
}

void IniParser::dumpToIniFile(FILE * fp) const
{
	iniparser_dump_ini(ini_, fp);
}

void IniParser::dump(FILE * fp) const
{
	iniparser_dump(ini_, fp);
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring IniParser::getStr(const std::wstring &key) const
#else
std::string IniParser::getStr(const std::string &key) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return String::mbs2wcs(iniparser_getstr(ini_, String::wcs2mbs(key).c_str()));
#else
	return iniparser_getstr(ini_, key.c_str());
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
std::wstring IniParser::getString(const std::wstring &key, const std::wstring &notfound) const
#else
std::string IniParser::getString(const std::string &key, const std::string &notfound) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return String::mbs2wcs(iniparser_getstring(ini_, String::wcs2mbs(key).c_str(), const_cast<char *>(String::wcs2mbs(notfound).c_str())));
#else
	return iniparser_getstring(ini_, key.c_str(), const_cast<char *>(notfound.c_str()));
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
int IniParser::getInt(const std::wstring &key, const int notfound) const
#else
int IniParser::getInt(const std::string &key, const int notfound) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return iniparser_getint(ini_, String::wcs2mbs(key).c_str(), notfound);
#else
	return iniparser_getint(ini_, key.c_str(), notfound);
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
double IniParser::getDouble(const std::wstring &key, const double notfound) const
#else
double IniParser::getDouble(const std::string &key, const double notfound) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return iniparser_getdouble(ini_, const_cast<char *>(String::wcs2mbs(key).c_str()), notfound);
#else
	return iniparser_getdouble(ini_, const_cast<char *>(key.c_str()), notfound);
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
bool IniParser::getBool(const std::wstring &key, const bool notfound) const
#else
bool IniParser::getBool(const std::string &key, const bool notfound) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return iniparser_getboolean(ini_, String::wcs2mbs(key).c_str(), notfound ? 1 : 0) == 1;
#else
	return iniparser_getboolean(ini_, key.c_str(), notfound ? 1 : 0) == 1;
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
bool IniParser::setStr(const std::wstring &entry, const std::wstring &val) const
#else
bool IniParser::setStr(const std::string &entry, const std::string &val) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return iniparser_setstr(ini_, const_cast<char *>(String::wcs2mbs(entry).c_str()), val.empty() ? NULL : const_cast<char *>(String::wcs2mbs(val).c_str())) == 0;
#else
	return iniparser_setstr(ini_, const_cast<char *>(entry.c_str()), val.empty() ? NULL : const_cast<char *>(val.c_str())) == 0;
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
void IniParser::unset(const std::wstring &entry) const
#else
void IniParser::unset(const std::string &entry) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	iniparser_unset(ini_, const_cast<char *>(String::wcs2mbs(entry).c_str()));
#else
	iniparser_unset(ini_, const_cast<char *>(entry.c_str()));
#endif
}

#if defined(UNICODE) || defined(_UNICODE)
bool IniParser::findEntry(const std::wstring &entry) const
#else
bool IniParser::findEntry(const std::string &entry) const
#endif
{
#if defined(UNICODE) || defined(_UNICODE)
	return iniparser_find_entry(ini_, const_cast<char *>(String::wcs2mbs(entry).c_str())) == 1;
#else
	return iniparser_find_entry(ini_, const_cast<char *>(entry.c_str())) == 1;
#endif
}

}  // namespace swl
