#if !defined(__SWL_BASE__LOG_EXCEPTION__H_)
#define __SWL_BASE__LOG_EXCEPTION__H_ 1


#include "swl/base/ExportBase.h"
#include <string>
#include <exception>
#include <iosfwd>


#if !defined(__FUNCTION__)
//#if defined(_UNICODE) || defined(UNICODE)
//#define __FUNCTION__ (L"")
//#else
#define __FUNCTION__ ("")
//#endif
#endif


namespace swl {

//-----------------------------------------------------------------------------------
// Exception for log.

/**
 *	@brief  Application ������ �߻��ϴ� log�� ���� exception class.
 *
 *	�� class�� C++ ǥ�� exception class�� std::exception���κ��� ��ӵǾ�����
 *	catch ������ �̿��Ͽ� exception handler�� �ۼ��ϸ� �ȴ�.
 *
 *	������ exception�� catch�Ͽ��� ���, �Ʒ��� ���� ������ �߻��� exception���κ��� �� �� �ִ�.
 *		- level
 *		- message
 *		- �߻���Ų ������ ���
 *		- �߻���Ų ���� �̸�
 *		- �� ��ȣ
 *		- class �̸�
 *		- �Լ� �̸�
 *
 *	Exception level�� ��� �⺻������ 7���� �������� ����������.
 *
 *	Exception�� �߻��� ��� �ش� ������ log stream�� ���� �ܺη� ����� �� �ִ�.
 *	setLogStream() �Լ��� resetLogStream() �Լ��� ����Ͽ� log�� ����� stream�� ���� �Ǵ� ������ �� �ִ�.
 */
class SWL_BASE_API LogException: public std::exception
{
public:
	/**
	 *	@brief  Base class�� type definition.
	 */
	typedef std::exception base_type;

public:
	/**
	 *	@brief  �⺻ log level.
	 *
	 *	class�� �����Ǿ� �ִ� �⺻ log level�� 7�ܰ�� 0���� 10���� level ���� ������.
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
	 *	@param[in]  level  exception�� log level ����.
	 *	@param[in]  message  exception message�� ����Ǵ� ��.
	 *	@param[in]  filePath  exception�� �߻��� ������ ��ü ���.
	 *	@param[in]  lineNo  exception�� �߻��� ������ �� ��ȣ.
	 *	@param[in]  methodName  exception�� �߻��� �Լ� �̸�. e.g.) class_name::method_name()�� ���¸� ���ϰ� �־�� �Ѵ�.
	 *
	 *	exception �߻��� �����Ǵ� ���� ���ڰ���κ��� log�� ���� �ʿ��� ������ �̾Ƴ��� �����Ѵ�.
	 */
	//LogException(const unsigned int level, const std::wstring &message, const std::wstring &filePath, const long lineNo, const std::wstring &methodName);
	LogException(const unsigned int level, const std::wstring &message, const std::string &filePath, const long lineNo, const std::string &methodName);
	//LogException(const unsigned int level, const std::string &message, const std::wstring &filePath, const long lineNo, const std::wstring &methodName);
	LogException(const unsigned int level, const std::string &message, const std::string &filePath, const long lineNo, const std::string &methodName);
	/**
	 *	@brief  [copy ctor] copy constructor.
	 *	@param[in]  rhs  ����Ǵ� ���� ��ü.
	 *
	 *	���ڷ� �Էµ� ��ü�� �����Ͽ� ���ο� ��ü�� �����Ѵ�..
	 */
	LogException(const LogException &rhs);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	�ش� class�κ��� �ڽ� class �Ļ��� �����ϵ��� virtual�� ����Ǿ� �ִ�.
	 */
	virtual ~LogException() throw();

private:
	LogException & operator=(const LogException &);

public:
	/**
	 *	@brief  �߻��� exception�� log level ���� ��ȯ.
	 *	@return  log level ���� unsigned int �� ������ ��ȯ.
	 */
	unsigned int getLevel() const  {  return level_;  }

	/**
	 *	@brief  �߻��� exception�� log message�� ��ȯ.
	 *	@return  log message�� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring & getMessage() const  {  return message_;  }
#else
	const std::string & getMessage() const  {  return message_;  }
#endif

	/**
	 *	@brief  exception�� �߻���Ų ������ ��ü ���.
	 *	@return  exception�� �߻��� ������ ��ü ��θ� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getFilePath() const  {  return filePath_;  }
#else
	std::string getFilePath() const  {  return filePath_;  }
#endif

	/**
	 *	@brief  exception�� �߻���Ų ���� �̸�.
	 *	@return  exception�� �߻��� ������ �̸��� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getFileName() const;
#else
	std::string getFileName() const;
#endif

	/**
	 *	@brief  exception�� �߻���Ų ������ �� ��ȣ.
	 *	@return  exception�� �߻��� ������ �� ��ȣ�� ��ȯ.
	 */
	long getLineNumber() const  {  return lineNo_;  }

	/**
	 *	@brief  exception�� �߻���Ų class�� �̸�.
	 *	@return  exception�� �߻��� class�� �̸��� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getClassName() const;
#else
	std::string getClassName() const;
#endif

	/**
	 *	@brief  exception�� �߻���Ų �Լ��� �̸�.
	 *	@return  exception�� �߻��� �Լ��� �̸��� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	std::wstring getMethodName() const;
#else
	std::string getMethodName() const;
#endif

	/**
	 *	@brief  �߻��� exception�� ������ ����ϱ� ���� ����� stream ��ü�� ����.
	 *	@param[in]  logStream  exception�� ������ ����� log stream ��ü.
	 *
	 *	exception�� �߻��� ��� �ش� ������ �ܺη� ����� log stream ��ü�� �����Ѵ�.
	 *
	 *	log stream�� ���� exception log�� ���������� ��µǱ� ���ؼ��� �Ʒ��� ������ �������Ѿ� �Ѵ�.
	 *		-# log stream ��ü�� ����ϴ� ���� �ش� ��ü�� valid�Ͽ��� �Ѵ�.
	 *		-# log stream ��ü�� open�� �����̾�� �Ѵ�.
	 *
	 */
#if defined(_UNICODE) || defined(UNICODE)
	static void setLogStream(std::wostream &logStream)
#else
	static void setLogStream(std::ostream &logStream)
#endif
	{  logStream_ = &logStream;  }
	/**
	 *	@brief  exception ������ �ܺη� ����ϱ� ���� stream ��ü�� ������ ����.
	 *
	 *	exception ������ �ܺη� ����ϴ� stream ��ü�� ������ �����Ѵ�.
	 */
	static void resetLogStream()
	{  logStream_ = NULL;  }

private:
	void report() const;

private:
	const unsigned int level_;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring message_;
#else
	const std::string message_;
#endif

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring filePath_;
#else
	const std::string filePath_;
#endif
	const long lineNo_;
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring methodName_;
#else
	const std::string methodName_;
#endif

#if defined(_UNICODE) || defined(UNICODE)
	static std::wostream *logStream_;
#else
	static std::ostream *logStream_;
#endif
};

}  // namespace swl


#endif  // __SWL_BASE__LOG_EXCEPTION__H_
