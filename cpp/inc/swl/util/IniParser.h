#if !defined(__SWL_UTIL__INI_PARSER__H_)
#define __SWL_UTIL__INI_PARSER__H_ 1


#include "swl/util/ExportUtil.h"
#include <string>


struct dictionary;

namespace swl {

//--------------------------------------------------------------------------
//

/**
 *	@brief  A parser class for interfacing an ini file
 *
 *	This class provides all the users with interface to read and write setting values from/into ini files
 */
class SWL_UTIL_API IniParser
{
public:
	/**
	 *	@brief  [ctor] Load a loadable ini file
	 *	@param[in]  iniFilePath  a full path of a loadable ini file
	 *	@throw  LogException  if a log exception occurred
	 *
	 *	This ctor reads some data from a loadable ini file.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	IniParser(const std::wstring &iniFilePath);
#else
	IniParser(const std::string &iniFilePath);
#endif
	/**
	 *	@brief  [dtor] Free the loaded ini file
	 *
	 *	This dtor releases resources that have been loaded from the loaded ini file.
	 */
	~IniParser();

private:
	IniParser(const IniParser &);
	IniParser & operator=(const IniParser &);
	
public:
	/**
	 *	@brief  Get number of sections in a dictionary
	 *	@return  int Number of sections found in dictionary
	 *
	 *	This function returns the number of sections found in a dictionary.
	 *	The test to recognize sections is done on the string stored in the
	 *	dictionary: a section name is given as "section" whereas a key is
	 *	stored as "section:key", thus the test looks for entries that do not
	 *	contain a colon.
	 *
	 *	This clearly fails in the case a section name contains a colon, but
	 *	this should simply be avoided.
	 *
	 *	This function returns -1 in case of error.
	 */
	int getSectionCount() const;

	/**
	 *	@brief  Get name for section n in a dictionary.
	 *	@param[in]  n  Section number (from 0 to nsec-1).
	 *	@return  string object containing a section name
	 *
	 *	This function locates the n-th section in a dictionary and returns
	 *	its name as a pointer to a string statically allocated inside the
	 *	dictionary. Do not free or modify the returned string!
	 *
	 *	This function returns NULL in case of error.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getSectionName(int n) const;
#else
	std::string getSectionName(int n) const;
#endif

	/**
	 *	@brief  Save a dictionary to a loadable ini file
	 *	@param[in]  fp  Opened file pointer to dump to
	 *	@return  void
	 *
	 *	This function dumps a given dictionary into a loadable ini file.
	 *	It is Ok to specify @c stderr or @c stdout as output files.
	 */
	void dumpToIniFile(FILE *fp) const;

	/**
	 *	@brief  Dump a dictionary to an opened file pointer.
	 *	@param[in]  fp  Opened file pointer to dump to.
	 *	@return  void
	 *
	 *	This function prints out the contents of a dictionary, one element by
	 *	line, onto the provided file pointer. It is OK to specify @c stderr
	 *	or @c stdout as output files. This function is meant for debugging
	 *	purposes mostly.
	 */
	void dump(FILE *fp) const;

	/**
	 *  @brief  Get the string associated to a key, return NULL if not found
	 *  @param[in]  key  Key string to look for
	 *  @return  string.
	 *
	 *  This function queries a dictionary for a key. A key as read from an
	 *  ini file is given as "section:key". If the key cannot be found,
	 *  NULL is returned.
	 *  The returned char pointer is pointing to a string allocated in
	 *  the dictionary, do not free or modify it.
	 *
	 *  This function is only provided for backwards compatibility with
	 *  previous versions of iniparser. It is recommended to use
	 *  iniparser_getstring() instead.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getStr(const std::wstring &key) const;
#else
	std::string getStr(const std::string &key) const;
#endif

	/**
	 *  @brief  Get the string associated to a key
	 *  @param[in]  key  Key string to look for
	 *  @param[in]  notfound  Default value to return if key not found.
	 *  @return  string
	 *
	 *  This function queries a dictionary for a key. A key as read from an
	 *  ini file is given as "section:key". If the key cannot be found,
	 *  the pointer passed as 'def' is returned.
	 *  The returned char pointer is pointing to a string allocated in
	 *  the dictionary, do not free or modify it.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getString(const std::wstring &key, const std::wstring &notfound) const;
#else
	std::string getString(const std::string &key, const std::string &notfound) const;
#endif

	/**
	 *  @brief  Get the string associated to a key, convert to an int
	 *  @param[in]  key  Key string to look for
	 *  @param[in]  notfound  Value to return in case of error
	 *  return  integer
	 *
	 *  This function queries a dictionary for a key. A key as read from an
	 *  ini file is given as "section:key". If the key cannot be found,
	 *  the notfound value is returned.
	 *
	 *  Supported values for integers include the usual C notation
	 *  so decimal, octal (starting with 0) and hexadecimal (starting with 0x)
	 *  are supported. Examples:
	 *
	 *  - "42"      ->  42
	 *  - "042"     ->  34 (octal -> decimal)
	 *  - "0x42"    ->  66 (hexa  -> decimal)
	 *
	 *  Warning: the conversion may overflow in various ways. Conversion is
	 *  totally outsourced to strtol(), see the associated man page for overflow
	 *  handling.
	 *
	 *  Credits: Thanks to A. Becker for suggesting strtol()
	 */
#if defined(_UNICODE) || defined(UNICODE)
	int getInt(const std::wstring &key, const int notfound) const;
#else
	int getInt(const std::string &key, const int notfound) const;
#endif

	/**
	 *  @brief  Get the string associated to a key, convert to a double
	 *  @param[in]  key  Key string to look for
	 *  @param[in]  notfound  Value to return in case of error
	 *  @return  double
	 *
	 *  This function queries a dictionary for a key. A key as read from an
	 *  ini file is given as "section:key". If the key cannot be found,
	 *  the notfound value is returned.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	double getDouble(const std::wstring &key, const double notfound) const;
#else
	double getDouble(const std::string &key, const double notfound) const;
#endif

	/**
	 *  @brief  Get the string associated to a key, convert to a boolean
	 *  @param[in]  key  Key string to look for
	 *  @param[in]  notfound  Value to return in case of error
	 *  @return  bool
	 *
	 *  This function queries a dictionary for a key. A key as read from an
	 *  ini file is given as "section:key". If the key cannot be found,
	 *  the notfound value is returned.
	 *
	 *  A true boolean is found if one of the following is matched:
	 *
	 *  - A string starting with 'y'
	 *  - A string starting with 'Y'
	 *  - A string starting with 't'
	 *  - A string starting with 'T'
	 *  - A string starting with '1'
	 *
	 *  A false boolean is found if one of the following is matched:
	 *
	 *  - A string starting with 'n'
	 *  - A string starting with 'N'
	 *  - A string starting with 'f'
	 *  - A string starting with 'F'
	 *  - A string starting with '0'
	 *
	 *  The notfound value returned if no boolean is identified, does not
	 *  necessarily have to be 0 or 1.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool getBool(const std::wstring &key, const bool notfound) const;
#else
	bool getBool(const std::string &key, const bool notfound) const;
#endif

	/**
	 *  @brief  Set an entry in a dictionary.
	 *  @param[in]  entry  Entry to modify (entry name)
	 *  @param[in]  val  New value to associate to the entry.
	 *  @return  bool true if Ok, false otherwise.
	 *
	 *  If the given entry can be found in the dictionary, it is modified to
	 *  contain the provided value. If it cannot be found, false is returned.
	 *  It is Ok to set val to NULL.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool setStr(const std::wstring &entry, const std::wstring &val) const;
#else
	bool setStr(const std::string &entry, const std::string &val) const;
#endif

	/**
	 *  @brief  Delete an entry in a dictionary
	 *  @param[in]  entry  Entry to delete (entry name)
	 *  @return  void
	 *
	 *  If the given entry can be found, it is deleted from the dictionary.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	void unset(const std::wstring &entry) const;
#else
	void unset(const std::string &entry) const;
#endif

	/**
	 *  @brief  Finds out if a given entry exists in a dictionary
	 *  @param[in]  entry  Name of the entry to look for
	 *  @return  bool true if entry exists, false otherwise
	 *
	 *  Finds out if a given entry exists in the dictionary. Since sections
	 *  are stored as keys with NULL associated values, this is the only way
	 *  of querying for the presence of sections in a dictionary.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	bool findEntry(const std::wstring &entry) const;
#else
	bool findEntry(const std::string &entry) const;
#endif

private:
	dictionary *ini_;
};

}  // namespace swl


#endif  // __SWL_UTIL__INI_PARSER__H_
