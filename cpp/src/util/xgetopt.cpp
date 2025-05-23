// XGetopt.cpp  Version 1.2
//
// Author:  Hans Dietrich
//          hdietrich2@hotmail.com
//
// Description:
//     XGetopt.cpp implements getopt(), a function to parse command lines.
//
// History
//     Version 1.2 - 2003 May 17
//     - Added Unicode support
//
//     Version 1.1 - 2002 March 10
//     - Added example to XGetopt.cpp module header
//
// This software is released into the public domain.
// You are free to use it in any way you like.
//
// This software is provided "as is" with no expressed
// or implied warranty.  I accept no liability for any
// damage or loss of business that this software may cause.
//
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// if you are using precompiled headers then include this line:
//#include "stdafx.h"
///////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
// if you are not using precompiled headers then include these lines:
//#include <windows.h>
//#include <stdio.h>
//#include <tchar.h>
///////////////////////////////////////////////////////////////////////////////


#include "swl/Config.h"
#include "xgetopt.h"
#include <cstdio>


///////////////////////////////////////////////////////////////////////////////
//
//  X G e t o p t . c p p
//
//
//  NAME
//       getopt -- parse command line options
//
//  SYNOPSIS
//       int getopt(int argc, char_t *argv[], char_t *optstring)
//
//       extern char_t *optarg;
//       extern int optind;
//
//  DESCRIPTION
//       The getopt() function parses the command line arguments. Its
//       arguments argc and argv are the argument count and array as
//       passed into the application on program invocation.  In the case
//       of Visual C++ programs, argc and argv are available via the
//       variables __argc and __argv (double underscores), respectively.
//       getopt returns the next option letter in argv that matches a
//       letter in optstring.  (Note:  Unicode programs should use
//       __targv instead of __argv.  Also, all character and string
//       literals should be enclosed in char_str( ) ).
//
//       optstring is a string of recognized option letters;  if a letter
//       is followed by a colon, the option is expected to have an argument
//       that may or may not be separated from it by white space.  optarg
//       is set to point to the start of the option argument on return from
//       getopt.
//
//       Option letters may be combined, e.g., "-ab" is equivalent to
//       "-a -b".  Option letters are case sensitive.
//
//       getopt places in the external variable optind the argv index
//       of the next argument to be processed.  optind is initialized
//       to 0 before the first call to getopt.
//
//       When all options have been processed (i.e., up to the first
//       non-option argument), getopt returns EOF, optarg will point
//       to the argument, and optind will be set to the argv index of
//       the argument.  If there are no non-option arguments, optarg
//       will be set to NULL.
//
//       The special option "--" may be used to delimit the end of the
//       options;  EOF will be returned, and "--" (and everything after it)
//       will be skipped.
//
//  RETURN VALUE
//       For option letters contained in the string optstring, getopt
//       will return the option letter.  getopt returns a question mark (?)
//       when it encounters an option letter not included in optstring.
//       EOF is returned when processing is finished.
//
//  BUGS
//       1)  Long options are not supported.
//       2)  The GNU double-colon extension is not supported.
//       3)  The environment variable POSIXLY_CORRECT is not supported.
//       4)  The + syntax is not supported.
//       5)  The automatic permutation of arguments is not supported.
//       6)  This implementation of getopt() returns EOF if an error is
//           encountered, instead of -1 as the latest standard requires.
//
//  EXAMPLE
//       BOOL CMyApp::ProcessCommandLine(int argc, char_t *argv[])
//       {
//           int c;
//
//           while ((c = getopt(argc, argv, char_str("aBn:"))) != EOF)
//           {
//               switch (c)
//               {
//                   case char_str('a'):
//                       TRACE(char_str("option a\n"));
//                       //
//                       // set some flag here
//                       //
//                       break;
//
//                   case char_str('B'):
//                       TRACE( char_str("option B\n"));
//                       //
//                       // set some other flag here
//                       //
//                       break;
//
//                   case char_str('n'):
//                       TRACE(char_str("option n: value=%d\n"), atoi(optarg));
//                       //
//                       // do something with value here
//                       //
//                       break;
//
//                   case char_str('?'):
//                       TRACE(char_str("ERROR: illegal option %s\n"), argv[optind-1]);
//                       return FALSE;
//                       break;
//
//                   default:
//                       TRACE(char_str("WARNING: no handler for option %c\n"), c);
//                       return FALSE;
//                       break;
//               }
//           }
//           //
//           // check for non-option args here
//           //
//           return TRUE;
//       }
//
///////////////////////////////////////////////////////////////////////////////
//--S [] 2015/06/05 : Sang-Wook Lee
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
//--E [] 2015/06/05 : Sang-Wook Lee
#include <tchar.h>
//--S [] 2015/06/05 : Sang-Wook Lee
#else
#if defined(_UNICODE) || defined(UNICODE)
#include <cwchar>
#else
#include <cstring>
#endif
#endif
//--E [] 2015/06/05 : Sang-Wook Lee

#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif

#if defined(_UNICODE) || defined(UNICODE)
#define char_t wchar_t
#define char_str(x) L##x
#else
#define char_t char
#define char_str(x) x
#endif

char_t	*optarg;		// global argument pointer
int		optind = 0; 	// global argv index

int getopt(int argc, char_t *argv[], char_t *optstring)
{
	static char_t *next = NULL;
	if (optind == 0)
		next = NULL;

	optarg = NULL;

	if (next == NULL || *next == char_str('\0'))
	{
		if (optind == 0)
			optind++;

		if (optind >= argc || argv[optind][0] != char_str('-') || argv[optind][1] == char_str('\0'))
		{
			optarg = NULL;
			if (optind < argc)
				optarg = argv[optind];
			return EOF;
		}

        //--S [] 2015/06/05 : Sang-Wook Lee
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
		//--E [] 2015/06/05 : Sang-Wook Lee
		if (_tcscmp(argv[optind], char_str("--")) == 0)
        //--S [] 2015/06/05 : Sang-Wook Lee
#else
#if defined(_UNICODE) || defined(UNICODE)
		if (wcscmp(argv[optind], char_str("--")) == 0)
#else
		if (strcmp(argv[optind], char_str("--")) == 0)
#endif
#endif
        //--E [] 2015/06/05 : Sang-Wook Lee
		{
			optind++;
			optarg = NULL;
			if (optind < argc)
				optarg = argv[optind];
			return EOF;
		}

		next = argv[optind];
		next++;		// skip past -
		optind++;
	}

	char_t c = *next++;
	//--S [] 2015/06/05 : Sang-Wook Lee
#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
	//--E [] 2015/06/05 : Sang-Wook Lee
	char_t *cp = _tcschr(optstring, c);
	//--S [] 2015/06/05 : Sang-Wook Lee
#else
#if defined(_UNICODE) || defined(UNICODE)
	char_t *cp = wcschr(optstring, c);
#else
	char_t *cp = strchr(optstring, c);
#endif
#endif
	//--E [] 2015/06/05 : Sang-Wook Lee

	if (cp == NULL || c == char_str(':'))
		return char_str('?');

	cp++;
	if (*cp == char_str(':'))
	{
		if (*next != char_str('\0'))
		{
			optarg = next;
			next = NULL;
		}
		else if (optind < argc)
		{
			optarg = argv[optind];
			optind++;
		}
		else
		{
			return char_str('?');
		}
	}

	return c;
}

#undef char_t
#undef char_str
