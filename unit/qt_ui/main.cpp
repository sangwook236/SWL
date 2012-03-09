#include <Qt/qapplication.h>
#include <cppunit/ui/qt/TestRunner.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <iostream>


int main(int argc, char **argv)
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

