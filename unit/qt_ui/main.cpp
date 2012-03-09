#include <qapplication.h>
#include <cppunit/ui/qt/TestRunner.h>
#include <cppunit/extensions/TestFactoryRegistry.h>


#if defined(UNICODE) || defined(_UNICODE)
int wmain(int argc, wchar_t **argv)
#else
int main(int argc, char **argv)
#endif
{
	try
	{
		QApplication app(argc, argv);

		//CppUnit::QtUi::TestRunner runner;
		CppUnit::QtTestRunner runner; 
		
		runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

		//runner.setOutputter();
		runner.run(true);
	}
	catch (const std::exception &e)
	{
		std::cout << "exception occurred !!!: " << e.what() << std::endl;
	}
	catch (...)
	{
		std::wcout << L"unknown exception occurred !!!" << std::endl;
	}

	std::wcout << L"press any key to exit ..." << std::endl;
	std::wcout.flush();
	std::wcin.get();

    return 0;
}

