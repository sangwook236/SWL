// XGetopt.h  Version 1.2
//
// Author:  Hans Dietrich
//          hdietrich2@hotmail.com
//
// This software is released into the public domain.
// You are free to use it in any way you like.
//
// This software is provided "as is" with no expressed
// or implied warranty.  I accept no liability for any
// damage or loss of business that this software may cause.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef XGETOPT_H
#define XGETOPT_H

#include "swl/util/ExportUtil.h"

extern SWL_UTIL_API int optind, opterr;

#if defined(UNICODE) || defined(_UNICODE)
extern SWL_UTIL_API wchar_t *optarg;

SWL_UTIL_API int getopt(int argc, wchar_t *argv[], wchar_t *optstring);
#else
extern SWL_UTIL_API char *optarg;

SWL_UTIL_API int getopt(int argc, char *argv[], char *optstring);
#endif

#endif //XGETOPT_H
