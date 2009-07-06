#include "swl/Config.h"
#include "AsyncEchoTcpSocketSession.h"
#include "EchoTcpSocketSession.h"
#include "TcpSocketServer.h"
#include "TcpSocketConnectionUsingSession.h"
#include <boost/thread.hpp>
#include <boost/smart_ptr.hpp>
#include <iostream>


#if defined(_DEBUG)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {

struct tcp_socket_worker_thread_functor
{
	void operator()()
	{
		boost::asio::io_service ioService;
		const unsigned short portNum_AsyncServer = 5005;
		const unsigned short portNum_HalfDuplex = 5006;

		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::AsyncEchoTcpSocketSession> > asyncServer(ioService, portNum_AsyncServer);
		swl::TcpSocketServer<swl::TcpSocketConnectionUsingSession<swl::EchoTcpSocketSession> > syncServer(ioService, portNum_HalfDuplex);

		std::cout << "start TCP socket servers: sync & async" << std::endl;
		ioService.run();
		std::cout << "teminate TCP socket servers: sync & async" << std::endl;
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
	boost::scoped_ptr<boost::thread> thrd(new boost::thread(tcp_socket_worker_thread_functor()));

	if (thrd.get())
		thrd->join();
	std::cout << "terminate thread for TCP socket servers" << std::endl;

	std::cout.flush();
	std::cin.get();
	return 0;
}
