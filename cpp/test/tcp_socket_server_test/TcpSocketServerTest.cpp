#include "swl/Config.h"
#include "EchoTcpSocketConnection.h"
#include "EchoTcpSocketSession.h"
#include "swl/util/TcpSocketConnectionUsingSession.h"
#include "swl/util/TcpSocketServer.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>
#include <cstdlib>

#if defined(_WIN64) || defined(WIN64) || defined(_WIN32) || defined(WIN32)
#include <vld/vld.h>
#endif


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

struct echo_tcp_socket_server_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;
		const unsigned short portNum_withoutSession = 6000;
		const unsigned short portNum_withSession = 7000;

		swl::TcpSocketServer<swl::EchoTcpSocketConnection> server(ioService, portNum_withoutSession);
		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::EchoTcpSocketSession> > sessionServer(ioService, portNum_withSession);

		std::cout << "start TCP socket servers: w/o & w/ session" << std::endl;
		ioService.run();
		std::cout << "finish TCP socket servers: w/o & w/ session" << std::endl;
	}
};

}  // namespace local
}  // unnamed namespace

int main(int argc, char *argv[])
{
	int retval = EXIT_SUCCESS;
	try
	{
		std::cout << "start thread for TCP socket servers" << std::endl;
		boost::scoped_ptr<boost::thread> thrd(new boost::thread(local::echo_tcp_socket_server_worker_thread_functor()));

		if (thrd.get())
			thrd->join();
		std::cout << "finish thread for TCP socket servers" << std::endl;
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
		std::cout << "Unknown exception caught" << std::endl;
		retval = EXIT_FAILURE;
	}

	//std::cout << "Press any key to exit ..." << std::endl;
	//std::cin.get();

	return retval;
}
