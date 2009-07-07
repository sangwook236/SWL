#include "swl/Config.h"
#include "EchoTcpSocketConnection.h"
#include "EchoTcpSocketSession.h"
#include "swl/util/TcpSocketConnectionUsingSession.h"
#include "swl/util/TcpSocketServer.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

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

}  // unnamed namespace

#if defined(_UNICODE) || defined(UNICODE)
int wmain()
#else
int main()
#endif
{
	std::cout << "start thread for TCP socket servers" << std::endl;
	boost::scoped_ptr<boost::thread> thrd(new boost::thread(echo_tcp_socket_server_worker_thread_functor()));

	if (thrd.get())
		thrd->join();
	std::cout << "finish thread for TCP socket servers" << std::endl;

	std::cout.flush();
	std::cin.get();
	return 0;
}
