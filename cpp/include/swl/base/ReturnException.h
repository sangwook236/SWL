#if !defined(__SWL_BASE__RETURN_EXCEPTION__H_)
#define __SWL_BASE__RETURN_EXCEPTION__H_ 1


#include "swl/base/ExportBase.h"
#include <boost/any.hpp>
#include <string>
#include <exception>


#if !defined(__FUNCTION__)
#if defined(_UNICODE) || defined(UNICODE)
#define __FUNCTION__ L""
#else
#define __FUNCTION__ ""
#endif
#endif


namespace swl {

//-----------------------------------------------------------------------------------
//	exception for function's return

/**
 *	@brief  application ������ �߻��ϴ� exception�� ���� class.
 *
 *	�� class�� C++ ǥ�� exception class�� std::exception���κ��� ��ӵǾ�����
 *	catch ������ �̿��Ͽ� exception handler�� �ۼ��ϸ� �ȴ�.
 *
 *	������ exception�� catch�Ͽ��� ���, �Ʒ��� ���� ������ �߻��� exception���κ��� �� �� �ִ�.
 *		- level
 *		- message
 *		- class �̸�
 *		- �Լ� �̸�
 *
 *	exception level�� ��� �⺻������ 7���� �������� ����������.
 */
class SWL_BASE_API ReturnException: public std::exception
{
public:
	/**
	 *	@brief  base class�� type definition.
	 */
	typedef std::exception base_type;

public:
	/**
	 *	@brief  �⺻ return level.
	 *
	 *	class�� �����Ǿ� �ִ� �⺻ return level�� 5�ܰ�� 0���� 10���� level ���� ������.
	 *		- L_INFO = 0
	 *		- L_WARN = 5
	 *		- L_ERROR = 8
	 *		- L_ASSERT = 8
	 *		- L_FATAL = 10
	 */
	enum { L_INFO = 0, L_WARN = 5, L_ERROR = 8, L_ASSERT = 8, L_FATAL = 10 };

public:
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  level  exception�� return level ����.
	 *	@param[in]  message  exception message�� ����Ǵ� ��.
	 *	@param[in]  methodName  exception�� �߻��� �Լ� �̸�. e.g.) class_name::method_name()�� ���¸� ���ϰ� �־�� �Ѵ�.
	 *
	 *	exception �߻��� �����Ǵ� ���� ���ڰ���κ��� return�� ���� �ʿ��� ������ �̾Ƴ��� �����Ѵ�.
	 */
	ReturnException(const unsigned int level, const std::wstring &message, const std::wstring &methodName);
	ReturnException(const unsigned int level, const std::wstring &message, const std::string &methodName);
	ReturnException(const unsigned int level, const std::string &message, const std::string &methodName);
	/**
	 *	@brief  [ctor] constructor.
	 *	@param[in]  level  exception�� return level ����.
	 *	@param[in]  returnVal  exception�� ����� return ���� ����Ǵ� ��.
	 *	@param[in]  message  exception message�� ����Ǵ� ��.
	 *	@param[in]  methodName  exception�� �߻��� �Լ� �̸�. e.g.) class_name::method_name()�� ���¸� ���ϰ� �־�� �Ѵ�.
	 *
	 *	exception �߻��� �����Ǵ� ���� ���ڰ���κ��� return�� ���� �ʿ��� ������ �̾Ƴ��� �����Ѵ�.
	 */
	ReturnException(const unsigned int level, const boost::any &returnVal, const std::wstring &message, const std::wstring &methodName);
	ReturnException(const unsigned int level, const boost::any &returnVal, const std::wstring &message, const std::string &methodName);
	ReturnException(const unsigned int level, const boost::any &returnVal, const std::string &message, const std::string &methodName);
	/**
	 *	@brief  [copy ctor] copy constructor.
	 *	@param[in]  rhs  ����Ǵ� ���� ��ü.
	 *
	 *	���ڷ� �Էµ� ��ü�� �����Ͽ� ���ο� ��ü�� �����Ѵ�..
	 */
	ReturnException(const ReturnException &rhs);
	/**
	 *	@brief  [dtor] virtual default destructor.
	 *
	 *	�ش� class�κ��� �ڽ� class �Ļ��� �����ϵ��� virtual�� ����Ǿ� �ִ�.
	 */
	virtual ~ReturnException() throw();

private:
	ReturnException & operator=(const ReturnException &);

public:
	/**
	 *	@brief  �߻��� exception�� return level ���� ��ȯ.
	 *	@return  return level ���� unsigned int �� ������ ��ȯ.
	 */
	unsigned int getLevel() const  {  return level_;  }

	/**
	 *	@brief  �߻��� exception�� ����� return ���� �����ϴ� ��� �̸� ��ȯ
	 *	@return  �߻��� exception�� return ���� ��ȯ.
	 */
	const boost::any & getReturnValue() const  {  return returnVal_;  }

	/**
	 *	@brief  �߻��� exception�� return message�� ��ȯ.
	 *	@return  return message�� ��ȯ.
	 *
	 *	unicode�� ����ϴ� ��� std::wstring����, �׷��� ���� ��� std::string ��ü�� ��ȯ�Ѵ�.
	 */
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring & getMessage() const  {  return message_;  }
#else
	const std::string & getMessage() const  {  return message_;  }
#endif

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

private:
	const unsigned int level_;

	boost::any returnVal_;

#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring message_;
#else
	const std::string message_;
#endif
#if defined(_UNICODE) || defined(UNICODE)
	const std::wstring methodName_;
#else
	const std::string methodName_;
#endif
};

}  // namespace swl


#endif  // __SWL_BASE__RETURN_EXCEPTION__H_
