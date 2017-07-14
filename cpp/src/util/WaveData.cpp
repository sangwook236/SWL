#include "swl/Config.h"
#include "swl/util/WaveData.h"
#include <iostream>


#if defined(_DEBUG) && defined(__SWL_CONFIG__USE_DEBUG_NEW)
#include "swl/ResourceLeakageCheck.h"
#define new DEBUG_NEW
#endif


namespace swl {

WaveData::WaveData()
: fp_(nullptr), isOpened_(false), header_(), headerSize_(0)
{
}

WaveData::~WaveData()
{
	closeWaveFile();
}

bool WaveData::openWaveFile(const std::string &filepath)
{
	closeWaveFile();

	// Load a wav file.
	fp_ = std::fopen(filepath.c_str(), "r");
	if (nullptr == fp_)
	{
		std::cerr << "Failed to open a wave file: " << filepath << std::endl;
		return false;
	}

	// Read the header.
	headerSize_ = std::fread(&header_, 1, sizeof(WaveHeader), fp_);
	if (headerSize_ > 0)
	{
		std::clog << "Header size = " << headerSize_ << " bytes." << std::endl;
		assert(1 == header_.audioFormat);
		isOpened_ = true;
		return true;
	}
	else
	{
		std::cerr << "Failed to open a wave file: " << filepath << std::endl;
		closeWaveFile();
		return false;
	}
}

void WaveData::closeWaveFile()
{
	if (fp_)
	{
		std::fclose(fp_);
		fp_ = nullptr;
	}
	isOpened_ = false;
}

void WaveData::rewind()
{
	if (isOpened())
		std::fseek(fp_, (long)headerSize_, SEEK_SET);
}

size_t WaveData::readRawData(std::vector<uint8_t> &data) const
{
	return isOpened() && !data.empty() ? std::fread(&data[0], sizeof(uint8_t), data.size(), fp_) : 0;
}

size_t WaveData::readRawData(const size_t start, std::vector<uint8_t> &data) const
{
	if (isOpened() && !data.empty())
	{
		std::fseek(fp_, long(headerSize_ + start), SEEK_SET);
		return std::fread(&data[0], sizeof(uint8_t), data.size(), fp_);
	}
	else return 0;
}

std::string WaveData::getChunkID() const
{
	return std::string(header_.chunkID, header_.chunkID + sizeof(header_.chunkID) / sizeof(header_.chunkID[0]));
}

std::string WaveData::getFormat() const
{
	return std::string(header_.format, header_.format + sizeof(header_.format) / sizeof(header_.format[0]));
}

std::string WaveData::getFmtHeader() const
{
	return std::string(header_.fmt, header_.fmt + sizeof(header_.fmt) / sizeof(header_.fmt[0]));
}

std::string WaveData::getDataHeader() const
{
	return std::string(header_.data, header_.data + sizeof(header_.data) / sizeof(header_.data[0]));
}

size_t WaveData::getFileSize() const
{
#if 1
	return headerSize_ + header_.dataSize;
#else
	std::fseek(fp_, 0, SEEK_END);

	const long fileSize = std::ftell(fp_);

	std::fseek(fp_, 0, SEEK_SET);

	return (size_t)fileSize;
#endif
}

}  // namespace swl
