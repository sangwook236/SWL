/************************************************************************/
/*                                                                      */
/*               Copyright 1998-2002 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */                
/*                                                                      */
/************************************************************************/
 
 
#if !defined(__SWL__DESIGN_BY_CONTRACT__H_)
#define __SWL__DESIGN_BY_CONTRACT__H_ 1

//#include "config.hxx"
#include <sstream>
#include <string>
#include <stdexcept>
          
/** \page ErrorReporting Error Reporting
    Exceptions and assertions provided by VIGRA

    <b>\#include</b> \<swl/DesignByContract.h\>
    
    VIGRA defines the following exception classes:
    
    \code
    namespace swl {
        class ContractViolation: public std::exception;
        class PreconditionViolation: public ContractViolation;
        class PostconditionViolation: public ContractViolation;
        class InvariantViolation: public ContractViolation;
    }
    \endcode
    
    The following associated macros throw the corresponding exception if 
    their PREDICATE evaluates to '<TT>false</TT>':
    
    \code
    SWL_PRECONDITION(PREDICATE, MESSAGE);
    SWL_POSTCONDITION(PREDICATE, MESSAGE);
    SWL_INVARIANT(PREDICATE, MESSAGE);
    \endcode
    
    The MESSAGE is passed to the exception and can be retrieved via
    the overloaded member function '<TT>exception.what()</TT>'. If the compiler
    flag '<TT>NDEBUG</TT>' is <em>not</em> defined, the file name and line number of 
    the error are automatically included in the message. The macro
    
    \code
    SWL_ASSERT(PREDICATE, MESSAGE);
    \endcode
    
    is identical to <tt>SWL_PRECONDITION()</tt> except that it is completely removed
    when '<TT>NDEBUG</TT>' is defined. This is useful for test that are only needed during 
    debugging, such as array index bound checking. The following macro
    
    \code
    SWL_FAIL(MESSAGE);
    \endcode
    
    unconditionally throws a '<TT>std::runtime_error</TT>' constructed from the message 
    (along with file name and line number, if NDEBUG is not set).
    
    <b> Usage:</b>
    
    Include-File:
    \<swl/DesignByContract.h\>
    <p>
    Namespace: swl (except for the macros, of course)
    
    \code
    int main(int argc, char ** argv)
    {
        try
        {
            const char* input_file_name = argv[1];

            // read input image
            swl::ImageImportInfo info(input_file_name);

            // fail if input image is not grayscale
            SWL_PRECONDITION(info.isGrayscale(), "Input image must be grayscale");

            ...// process image
        }
        catch (std::exception & e)
        {
            std::cerr << e.what() << std::endl;  // print message
            return 1;
        }

        return 0;
    }
    \endcode
**/

namespace swl {

class ContractViolation: public std::exception
{
public:
	ContractViolation()
	{}

	ContractViolation(char const *prefix, char const *message, char const *file, const int line)
	{
		(*this) << "\n" << prefix << "\n" << message << "\n(" << file << ":" << line << ")\n";
	}

	ContractViolation(char const *prefix, char const *message)
	{
		(*this) << "\n" << prefix << "\n" << message << "\n";
	}

	~ContractViolation() throw()
	{}

	template<class T>
	ContractViolation & operator<<(T const &data)
	{
		std::ostringstream what;
		what << data;
		what_ += what.str();
		return *this;
	}

	virtual const char * what() const throw()
	{
		try
		{
			return what_.c_str();
		}
		catch (...)
		{
			return "swl::ContractViolation: error message was lost, sorry.";
		}
	}

private:
	std::string what_;
};

class PreconditionViolation: public ContractViolation
{
public:
	PreconditionViolation(char const *message, const char *file, const int line)
	: ContractViolation("Precondition violation!", message, file, line)
	{}

	PreconditionViolation(char const *message)
	: ContractViolation("Precondition violation!", message)
	{}
};

class PostconditionViolation: public ContractViolation
{
public:
	PostconditionViolation(char const *message, const char *file, const int line)
	: ContractViolation("Postcondition violation!", message, file, line)
	{}

	PostconditionViolation(char const *message)
	: ContractViolation("Postcondition violation!", message)
	{}
};

class InvariantViolation: public ContractViolation
{
public:
	InvariantViolation(char const *message, const char *file, const int line)
	: ContractViolation("Invariant violation!", message, file, line)
	{}

	InvariantViolation(char const *message)
	: ContractViolation("Invariant violation!", message)
	{}
};

//#if !defined(NDEBUG)
#if 1

inline void throwInvariantError(const bool predicate, char const *message, char const *file, const int line)
{
	if (!predicate) throw swl::InvariantViolation(message, file, line); 
}

inline void throwInvariantError(const bool predicate, const std::string &message, char const *file, const int line)
{
	if (!predicate) throw swl::InvariantViolation(message.c_str(), file, line); 
}

inline void throwPreconditionError(const bool predicate, char const *message, char const *file, const int line)
{
	if (!predicate) throw swl::PreconditionViolation(message, file, line); 
}

inline void throwPreconditionError(const bool predicate, const std::string &message, char const *file, const int line)
{
	if (!predicate) throw swl::PreconditionViolation(message.c_str(), file, line); 
}

inline void throwPostconditionError(const bool predicate, char const *message, char const *file, const int line)
{
	if (!predicate) throw swl::PostconditionViolation(message, file, line); 
}

inline void throwPostconditionError(const bool predicate, const std::string &message, char const *file, const int line)
{
	if (!predicate) throw swl::PostconditionViolation(message.c_str(), file, line); 
}

inline void throwRuntimeError(char const *message, char const *file, const int line)
{
	std::ostringstream what;
	what << "\n" << message << "\n(" << file << ":" << line << ")\n";
	throw std::runtime_error(what.str()); 
}

inline void throwRuntimeError(const std::string &message, char const *file, const int line)
{
	std::ostringstream what;
	what << "\n" << message << "\n(" << file << ":" << line << ")\n";
	throw std::runtime_error(what.str()); 
}

#define SWL_PRECONDITION(PREDICATE, MESSAGE) swl::throwPreconditionError((PREDICATE), (MESSAGE), __FILE__, __LINE__)

#define SWL_ASSERT(PREDICATE, MESSAGE) SWL_PRECONDITION((PREDICATE), (MESSAGE))

#define SWL_POSTCONDITION(PREDICATE, MESSAGE) swl::throwPostconditionError((PREDICATE), (MESSAGE), __FILE__, __LINE__)

#define SWL_INVARIANT(PREDICATE, MESSAGE) swl::throwInvariantError((PREDICATE), (MESSAGE), __FILE__, __LINE__)

#define SWL_FAIL(MESSAGE) swl::throwRuntimeError((MESSAGE), __FILE__, __LINE__)

#else  // NDEBUG

inline void throwInvariantError(const bool predicate, char const *message)
{
	if (!predicate) throw swl::InvariantViolation(message); 
}

inline void throwPreconditionError(const bool predicate, char const *message)
{
	if (!predicate) throw swl::PreconditionViolation(message); 
}

inline void throwPostconditionError(const bool predicate, char const *message)
{
	if (!predicate) throw swl::PostconditionViolation(message); 
}

inline void throwInvariantError(const bool predicate, const std::string &message)
{
	if (!predicate) throw swl::InvariantViolation(message.c_str()); 
}

inline void throwPreconditionError(const bool predicate, const std::string &message)
{
	if (!predicate) throw swl::PreconditionViolation(message.c_str()); 
}

inline void throwPostconditionError(const bool predicate, const std::string &message)
{
	if (!predicate) throw swl::PostconditionViolation(message.c_str()); 
}

#define SWL_PRECONDITION(PREDICATE, MESSAGE) swl::throwPreconditionError((PREDICATE), (MESSAGE))

#define SWL_ASSERT(PREDICATE, MESSAGE)

#define SWL_POSTCONDITION(PREDICATE, MESSAGE) swl::throwPostconditionError((PREDICATE), (MESSAGE))

#define SWL_INVARIANT(PREDICATE, MESSAGE) swl::throwInvariantError((PREDICATE), (MESSAGE))

#define SWL_FAIL(MESSAGE) throw std::runtime_error((MESSAGE))

#endif  // NDEBUG

}  // namespace swl


#endif  // __SWL__DESIGN_BY_CONTRACT__H_
