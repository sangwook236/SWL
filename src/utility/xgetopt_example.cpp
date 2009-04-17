#include "xgetopt.h"

#if defined(WIN32) && defined(_DEBUG)
void* __cdecl operator new(size_t nSize, const char* lpszFileName, int nLine);
#define new new(__FILE__, __LINE__)
//#pragma comment(lib, "mfc80ud.lib")
#endif

int main(int argc, char * argv[])
{
	// "xgetopt_example -ab -c -C -d foo -e123 xyz"

	while ((c = getopt(argc, argv, _T("abcCd:e:f"))) != EOF)
	{
		switch (c)
		{
		case _T('a'):
			break;
		case _T('b'):
			break;
		case _T('c'):
			break;
		case _T('C'):
			break;
		case _T('d'):
			break;
		case _T('e'):
			break;
		case _T('?'):
			break;
		default:
			break;
		}
	}
}
