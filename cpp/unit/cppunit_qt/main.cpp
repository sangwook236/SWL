#if defined(WIN32)
#include <vld/vld.h>
#endif
#include <Qt/qapplication.h>
#include <cppunit/ui/qt/TestRunner.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <iostream>
#include <cstdlib>


int main(int argc, char **argv)
{
	int retval = EXIT_SUCCESS;
	try
	{
		QApplication app(argc, argv);

		//CppUnit::QtUi::TestRunner runner;
		CppUnit::QtTestRunner runner;

		runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());

		//runner.setOutputter();
		runner.run(true);
	}
    catch (const std::bad_alloc &e)
	{
		std::cout << "std::bad_alloc caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (const std::exception &e)
	{
		std::cout << "std::exception caught: " << e.what() << std::endl;
		retval = EXIT_FAILURE;
	}
	catch (...)
	{
		std::cout << "unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	//std::cout << "press any key to exit ..." << std::endl;
	//std::cin.get();

    return retval;
}

