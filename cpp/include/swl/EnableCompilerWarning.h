#if !defined(__SWL__ENABLE_COMPILER_WARNING__H_)
#define __SWL__ENABLE_COMPILER_WARNING__H_ 1


#if defined(_MSC_VER)

#if (_MSC_VER < 1200)  // VC5 and before
#	pragma warning(default: 4018)  // signed/unsigned mismatch
#	pragma warning(default: 4290)  // c++ exception specification ignored
#	pragma warning(default: 4389)  // '==' : signed/unsigned mismatch
#	pragma warning(default: 4610)  // struct '...' can never be instantiated - user defined constructor required
#endif

#if (_MSC_VER < 1300)  // VC6/eVC4 
#	pragma warning(default: 4097)  // typedef-name used as based class of (...)
#	pragma warning(default: 4251)  // DLL interface needed
#	pragma warning(default: 4284)  // for -> operator
#	pragma warning(default: 4503)  // decorated name length exceeded, name was truncated
#	pragma warning(default: 4514)  // unreferenced inline function has been removed
#	pragma warning(default: 4660)  // template-class specialization '...' is already instantiated
#	pragma warning(default: 4701)  // local variable 'base' may be used without having been initialized
#	pragma warning(default: 4710)  // function (...) not inlined
#	pragma warning(default: 4786)  // identifier truncated to 255 characters
#endif

#if (_MSC_VER <= 1310)
#	pragma warning(default: 4511)  // copy constructor cannot be generated
#endif

#if (_MSC_VER < 1300) && defined(UNDER_CE)
#	pragma warning(default: 4201)  // nonstandard extension used : nameless struct/union
#	pragma warning(default: 4214)  // nonstandard extension used : bit field types other than int
#endif

#pragma warning(default: 4075)  // initializers put in unrecognized initialization area
// This warning is default only for the c_locale_win32.c file compilation:
#pragma warning(default: 4100)  // unreferenced formal parameter
#pragma warning(default: 4127)  // conditional expression is constant
#pragma warning(default: 4146)  // unary minus applied to unsigned type
#pragma warning(default: 4245)  // conversion from 'enum ' to 'unsigned int', signed/unsigned mismatch
#pragma warning(default: 4244)  // implicit conversion: possible loss of data
#pragma warning(default: 4512)  // assignment operator could not be generated
#pragma warning(default: 4571)  // catch(...) blocks compiled with /EHs do not catch or re-throw Structured Exceptions
#pragma warning(default: 4702)  // unreachable code (appears in release with warning level4)

// dums: This warning, signaling deprecated C functions like strncpy,
// will have to be fixed one day:
#pragma warning(default: 4996)

#pragma warning(pop)

#endif


#endif  // __SWL__ENABLE_COMPILER_WARNING__H_
