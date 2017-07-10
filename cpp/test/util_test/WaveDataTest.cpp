#include "swl/Config.h"
#include "swl/util/WaveData.h"
#include <boost/timer/timer.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <memory>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace {
namespace local {

}  // namespace local
}  // unnamed namespace

void wave_data()
{
	const size_t BUFFER_SIZE = 4096;

	const std::string wav_filename("./data/util/test1.wav");
	swl::WaveData wav;

	// Open a wave file.
	if (wav.openWaveFile(wav_filename))
	{
		// Chunk descriptor.
		std::cout << "Chunk ID = " << wav.getChunkID() << std::endl;
		std::cout << "Chunk size = " << wav.getChunkSize() << " bytes." << std::endl;
		std::cout << "Format = " << wav.getFormat() << std::endl;

		// "fmt" sub-chunk (8 + fmtSize).
		std::cout << "FMT header = " << wav.getFmtHeader() << std::endl;
		std::cout << "FMT size = " << wav.getFmtSize() << " bytes." << std::endl;
		std::cout << "Audio format = " << wav.getAudioFormat() << std::endl;
		std::cout << "Number of channels = " << wav.getNumberOfChannels() << std::endl;
		std::cout << "Sampling rate = " << wav.getSamplesPerSecond() << std::endl;
		std::cout << "Bytes per second = " << wav.getBytesPerSecond() << std::endl;
		std::cout << "Block align = " << wav.getBlockAlign() << std::endl;
		std::cout << "Bits per sample = " << wav.getBitsPerSample() << std::endl;

		// "data" sub-chunk (8 + dataSize).
		std::cout << "DATA header = " << wav.getDataHeader() << std::endl;
		std::cout << "Data size = " << wav.getDataSize() << " bytes." << std::endl;

		//
		std::cout << "Number of samples = " << wav.getNumberOfSamples() << std::endl;
		std::cout << "Actual file size = " << wav.getFileSize() << " bytes." << std::endl;

		// Read the raw data.
		{
			std::cout << "Read the data: " << std::endl;
			size_t bytesRead, totalBytesRead = 0;
#if 1
			std::vector<uint8_t> wav_data(BUFFER_SIZE);
			while ((bytesRead = wav.readRawData(wav_data)) > 0)
#else
			std::vector<uint8_t> wav_data(wav.getDataSize());
			while ((bytesRead = wav.readRawData(wav_data)) > 0)
#endif
			{
				// Do something.
				totalBytesRead += bytesRead;

				// FIXME [check] >> Bytes read is incorrect.
				std::cout << '\t' << bytesRead << " bytes read." << std::endl;
			}

			std::cout << "\tTotal " << totalBytesRead << " bytes read." << std::endl;

#if 1
			// Display.
			//for (size_t idx = 0; idx < totalBytesRead; ++idx)
			for (size_t idx = 0; idx < 80; ++idx)
				std::cout << std::hex << (int)wav_data[idx] << ", ";
			std::cout << std::dec << std::endl;
#endif
		}

		// Read all the channel data.
		{
			std::cout << "Read all the channel data: " << std::endl;

			const size_t &numSamples = wav.getNumberOfSamples();
			std::vector<std::vector<int16_t> > allChannelData(wav.getNumberOfChannels());
			for (size_t ch = 0; ch < wav.getNumberOfChannels(); ++ch)
				allChannelData[ch].resize(numSamples);

			size_t numSamplesRead;
			{
				boost::timer::auto_cpu_timer timer;
				if ((numSamplesRead = wav.readAllChannelData(0, allChannelData)) > 0)
				{
					// Do something.
				}
			}

			std::cout << '\t' << numSamplesRead << " samples (" << numSamplesRead * wav.getBlockAlign() << " bytes) read." << std::endl;

#if 1
			// Display.
			//for (size_t idx = 0; idx < numSamplesRead; ++idx)
			for (size_t idx = 0; idx < 20; ++idx)
				for (uint16_t ch = 0; ch < wav.getNumberOfChannels(); ++ch)
					std::cout << std::hex << allChannelData[ch][idx] << ", ";
			std::cout << std::dec << std::endl;
#endif
		}

		// Read a single channel data.
		{
			std::cout << "Read a single channel data: " << std::endl;

			const size_t channel = 0;

			const size_t &numSamples = wav.getNumberOfSamples();
			std::vector<int16_t> channelData(numSamples);
			size_t numSamplesRead;
			{
				boost::timer::auto_cpu_timer timer;
				if ((numSamplesRead = wav.readChannelData(channel, 0, channelData)) > 0)
				{
					// Do something.
				}
			}

			std::cout << '\t' << numSamplesRead << " samples (" << numSamplesRead * wav.getBlockAlign() << " bytes) read." << std::endl;

#if 1
			// Display.
			//for (size_t idx = 0; idx < numSamplesRead; ++idx)
			for (size_t idx = 0; idx < 20; ++idx)
				std::cout << std::hex << channelData[idx] << ", ";
			std::cout << std::dec << std::endl;
#endif
		}
	}
	else
		std::cerr << "Failed to open a wav file: " << wav_filename << std::endl;
}
