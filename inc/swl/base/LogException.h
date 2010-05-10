#if !defined(__SWL_BASE__LOG_EXCEPTION__H_)
#define __SWL_BASE__LOG_EXCEPTION__H_ 1


#include "swl/base/ExportBase.h"
#include <string>
#include <exception>
#include <iosfwd>


#if !defined(__FUNCTION__)
#if defined(UNICODE) || defined(_UNICODE)
#define __FUNCTION__ L""
#else
#define __FUNCTION__ ""
#endif
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for log

/**
 *	@brief  application 내에서 발생하는 exception을 위한 class.
 *
 *	본 class는 C++ 표준 exception class인 std::exception으로부터 상속되었으며
 *	catch 구문을 이용하여 exception handler를 작성하면 된다.
 *
 *	던져진 exception을 catch하였을 경우, 아래와 같은 정보를 발생한 exception으로부터 알 수 있다.
 *		- level
 *		- message
 *		- 발생시킨 파일의 경로
 *		- 발생시킨 파일 이름
 *		- 줄 번호
 *		- class 이름
 *		- 함수 이름
 *
 *	exception level의 경우 기본적으로 7가지 수준으로 나뉘어진다.
 *
 *	exception이 발생한 경우 해당 내용을 log stream을 통해 외부로 출력할 수 있다.
 *	setLogStream() 함수와 resetLogStream() 함수를 사용하여 log를 출력할 stream을 설정 또는 해제할 수 있다.
 */
class SWL_BASE_API LogException: public std::exception
{
public:
	/**
	 *	@brief  base class의 type definition.
	 */
	typedef std::exception base_type;

public:
	/**
	 *	@brief  기본 log level.
	 *
	 *	class에 설정되어 있는 기본 log level은 7단계로 0에서 10까지 level 값을 가진다.
	 *		- L_DEBUG = 0
	 *		- L_TRACE = 2
	 *		- L_INFO = 4
	 *		- L_WARN = 6
	 *		- L_ERROR = 8
	 *		- L_ASSERT = 8
	 *		- L_FATAL = 10
	 */
	enum { L_DEBUG = 0, L_TRACE = 2, L_INFO = 4, L_WARN = 6, L_ERROR = 8, L_ASSERT = 8, L_FATAL = 10 };

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  level  exception의 log level 수준.
	 *	@param[in]  message  exception message가 저장되는 곳.
	 *	@param[in]  filePath  exception이 발생한 파일의 전체 경로.
	 *	@param[in]  lineNo  exception이 발생한 파일의 줄 번호.
	 *	@param[in]  methodName  exception이 발생한 함수 이름. e.g.) class_name::method_name()의 형태를 취하고 있어야 한다.
	 *
	 *	exception 발생시 지정되는 여러 인자값들로부터 log를 위해 필요한 정보를 뽑아내고 관리한다.
	 */
	LogException(const unsigned int level, const std::wstring &message, const std::wstring &filePath, const long lineNo, const std::wstring &methodName);
	LogException(const unsigned int level, const std::wstring &message, const std::string &filePath, const long lineNo, const std::string &methodName);
	LogException(const unsigned int level, const std::string &message, const std::string &filePath, const long lineNo, const std::string &methodName);
	/**
	 *	@brief  [copy ctor] copy constructor.
	 *	@param[in]  rhs  복사되는 원본 객체.
	 *
	 *	인자로 입력된 객체를 복사하여 새로운 객체를 생성한다..
	 */
	LogException(const LogException &rhs);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	해당 class로부터 자식 class 파생이 가능하도록 virtual로 선언되어 있다.
	 */
	virtual ~LogException() throw();

private:
	LogException & operator=(const LogException &);

public:
	/**
	 *	@brief  발생한 exception의 log level 값을 반환.
	 *	@return  log level 값이 unsigned int 형 값으로 반환.
	 */
	unsigned int getLevel() const  {  return level_;  }

	/**
	 *	@brief  발생한 exception의 log message를 반환.
	 *	@return  log message를 반환.
	 *
	 *	unicode를 사용하는 경우 std::wstring형을, 그렇지 않은 경우 std::string 객체를 반환한다.
	 */
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring & getMessage() const  {  return message_;  }
#else
	const std::string & getMessage() const  {  return message_;  }
#endif

	/**
	 *	@brief  exception을 발생시킨 파일의 전체 경로.
	 *	@return  exception이 발생한 파일의 전체 경로를 반환.
	 *
	 *	unicode를 사용하는 경우 std::wstring형을, 그렇지 않은 경우 std::string 객체를 반환한다.
	 */
#if defined(UNICODE) || defined(_UNICODE)
	std::wstring getFilePath() const  {  return filePath_;  }
#else
	std::string getFilePath() const  {  return filePath_;  }
#endif

	/**
	 *	@brief  exception을 발생시킨 파일 이름.
	 *	@return  exception이 발생한 파일의 이름을 반환.
	 *
	 *	unicode를 사용하는 경우 std::wstring형을, 그렇지 않은 경우 std::string 객체를 반환한다.
	 */
#if defined(UNICODE) || defined(_UNICODE)
	std::wstring getFileName() const;
#else
	std::string getFileName() const;
#endif

	/**
	 *	@brief  exception을 발생시킨 파일의 줄 번호.
	 *	@return  exception이 발생한 파일의 줄 번호를 반환.
	 */
	long getLineNumber() const  {  return lineNo_;  }

	/**
	 *	@brief  exception을 발생시킨 class의 이름.
	 *	@return  exception이 발생한 class의 이름을 반환.
	 *
	 *	unicode를 사용하는 경우 std::wstring형을, 그렇지 않은 경우 std::string 객체를 반환한다.
	 */
#if defined(UNICODE) || defined(_UNICODE)
	std::wstring getClassName() const;
#else
	std::string getClassName() const;
#endif

	/**
	 *	@brief  exception을 발생시킨 함수의 이름.
	 *	@return  exception이 발생한 함수의 이름을 반환.
	 *
	 *	unicode를 사용하는 경우 std::wstring형을, 그렇지 않은 경우 std::string 객체를 반환한다.
	 */
#if defined(UNICODE) || defined(_UNICODE)
	std::wstring getMethodName() const;
#else
	std::string getMethodName() const;
#endif

	/**
	 *	@brief  발생한 exception의 내용을 출력하기 위해 사용할 stream 객체를 설정.
	 *	@param[in]  logStream  exception의 내용을 출력한 log stream 객체.
	 *
	 *	exception이 발생한 경우 해당 내용을 외부로 출력할 log stream 객체를 설정한다.
	 *
	 *	log stream를 통해 exception log가 정상적으로 출력되기 위해서는 아래의 조건을 만족시켜야 한다.
	 *		-# log stream 객체를 사용하는 동안 해당 객체는 valid하여야 한다.
	 *		-# log stream 객체는 open된 상태이어야 한다.
	 *	
	 */
#if defined(UNICODE) || defined(_UNICODE)
	static void setLogStream(std::wostream &logStream)
#else
	static void setLogStream(std::ostream &logStream)
#endif
	{  logStream_ = &logStream;  }
	/**
	 *	@brief  exception 내용을 외부로 출력하기 위한 stream 객체의 설정을 해제.
	 *
	 *	exception 내용을 외부로 출력하는 stream 객체의 설정을 해제한다.
	 */
	static void resetLogStream()
	{  logStream_ = NULL;  }

private:
	void report() const;

private:
	const unsigned int level_;

#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring message_;
#else
	const std::string message_;
#endif

#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring filePath_;
#else
	const std::string filePath_;
#endif
	const long lineNo_;
#if defined(UNICODE) || defined(_UNICODE)
	const std::wstring methodName_;
#else
	const std::string methodName_;
#endif

#if defined(UNICODE) || defined(_UNICODE)
	static std::wostream *logStream_;
#else
	static std::ostream *logStream_;
#endif
};

}  // namespace swl


#endif  // __SWL_BASE__LOG_EXCEPTION__H_
