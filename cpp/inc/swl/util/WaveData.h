#if !defined(__SWL_UTIL__WAVE_DATA__H_ )
#define __SWL_UTIL__WAVE_DATA__H_ 1


#include "swl/util/ExportUtil.h"
#include <string>
#include <vector>
#include <cstdio>


namespace swl {

// Interfacing class for wave file.
class SWL_UTIL_API WaveData final
{
private:
	// REF [site] >> http://soundfile.sapp.org/doc/WaveFormat
	struct WaveHeader
	{
		// Chunk descriptor.
		uint8_t chunkID[4];  // RIFF header: "RIFF".
		uint32_t chunkSize;  // RIFF chunk size [byte]: chunkSize = 36 + dataSize = 4 + (8 + fmtSize) + (8 + dataSize). This is the size of the rest of the chunk following this number.
		uint8_t format[4];  // WAVE header: "WAVE".
		// "fmt" sub-chunk (8 + fmtSize).
		uint8_t fmt[4];  // FMT header: "fmt ".
		uint32_t fmtSize;  // fmt chunk size [byte]. 16 for PCM. This is the size of the rest of the Subchunk which follows this number.
		uint16_t audioFormat;  // Audio format: PCM=1, mulaw=6, alaw=7, IBM Mu-Law=257, IBM A-Law=258, ADPCM=259. Values other than 1 indicate some form of compression.
		uint16_t channels;  // Number of channels: Mono=1, Stereo=2.
		uint32_t samplesPerSec;  // Sampling frequency/rate [Hz].
		uint32_t bytesPerSec;  // Bytes rate. bytesPerSec = samplesPerSec * channels * bitsPerSample/8.
		uint16_t blockAlign;  // 16-bit mono=2, 16-bit stereo=4. The number of bytes for one sample including all channels. blockAlign = channels * bitsPerSample/8.
		uint16_t bitsPerSample;  // Bits per sample.
		// "data" sub-chunk (8 + dataSize).
		uint8_t data[4];  // "data".
		uint32_t dataSize;  // Data size [byte].
	};

public:
	WaveData();
	~WaveData();

private:
	WaveData(const WaveData &) = delete;
	WaveData & operator=(const WaveData &) = delete;

public:
	bool openWaveFile(const std::string &filepath);
	void closeWaveFile();

	void rewind();
	size_t readRawData(std::vector<uint8_t> &data) const;
	size_t readRawData(const size_t start, std::vector<uint8_t> &data) const;

	template<typename T>
	size_t readAllChannelData(const size_t sampleStartIdx, std::vector<std::vector<T> > &allChannelData) const;
	template<typename T>
	size_t readChannelData(const size_t channel, const size_t sampleStartIdx, std::vector<T> &channelData) const;

	bool isOpened() const {  return fp_ && isOpened_;  }

	// Header info.
	std::string getChunkID() const;
	uint32_t getChunkSize() const { return header_.chunkSize; }
	std::string getFormat() const;
	std::string getFmtHeader() const;
	uint32_t getFmtSize() const { return header_.fmtSize; }
	uint16_t getAudioFormat() const { return header_.audioFormat; }
	uint16_t getNumberOfChannels() const { return header_.channels; }
	uint32_t getSamplesPerSecond() const { return header_.samplesPerSec; }
	uint32_t getBytesPerSecond() const { return header_.bytesPerSec; }
	uint16_t getBlockAlign() const { return header_.blockAlign; }
	uint16_t getBitsPerSample() const { return header_.bitsPerSample; }
	std::string getDataHeader() const;
	uint32_t getDataSize() const { return header_.dataSize; }

	//
	size_t getBytesPerSample() const {  return getBitsPerSample() / 8;  }
	size_t getNumberOfSamples() const
	{
		//return getChunkSize() / getBytesPerSample();
		//return getDataSize() / (getBytesPerSample() * getNumberOfChannels());
		return getDataSize() / getBlockAlign();
	}
	size_t getFileSize() const;

private:
	FILE *fp_;
	bool isOpened_;

	WaveHeader header_;
	size_t headerSize_;
};

template<typename T>
size_t WaveData::readAllChannelData(const size_t sampleStartIdx, std::vector<std::vector<T> > &allChannelData) const
{
	if (!isOpened() || allChannelData.empty() || allChannelData.size() != getNumberOfChannels()) return 0;

	const size_t &sampleCount = allChannelData[0].size();
	for (size_t ch = 1; ch < getNumberOfChannels(); ++ch)
		if (allChannelData[ch].size() != sampleCount) return 0;

	std::fseek(fp_, long(headerSize_ + sampleStartIdx * getBlockAlign()), SEEK_SET);
#if 1
	T sample;
	size_t samplesRead;
	for (size_t idx = 0; idx < sampleCount; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
		{
			samplesRead = std::fread(&sample, getBytesPerSample(), 1, fp_);
			//samplesRead = std::fread(&sample, sizeof(T), 1, fp_);
			if (1 == samplesRead)
				allChannelData[ch][idx] = sample;
			else return idx;
		}

	return sampleCount;
#elif 0
	std::vector<T> sample(getNumberOfChannels());
	size_t samplesRead;
	for (size_t idx = 0; idx < sampleCount; ++idx)
	{
		samplesRead = std::fread(&sample[0], getBytesPerSample(), getNumberOfChannels(), fp_);
		//samplesRead = std::fread(&sample[0], sizeof(T), getNumberOfChannels(), fp_);
		if (getNumberOfChannels() == samplesRead)
			for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
				allChannelData[ch][idx] = sample[ch];
		else return idx;
	}

	return sampleCount;
#else
	std::vector<T> sample(getNumberOfChannels() * sampleCount);
	const size_t samplesRead = std::fread(&sample[0], getBytesPerSample(), getNumberOfChannels() * sampleCount, fp_) / getNumberOfChannels();
	//const size_t samplesRead = std::fread(&sample[0], sizeof(T), getNumberOfChannels() * sampleCount, fp_);
	for (size_t idx = 0; idx < samplesRead; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
			allChannelData[ch][idx] = sample[idx * getNumberOfChannels() + ch];

	return samplesRead;
#endif
}

template<typename T>
size_t WaveData::readChannelData(const size_t channel, const size_t sampleStartIdx, std::vector<T> &channelData) const
{
	if (!isOpened() || channel >= getNumberOfChannels()) return 0;

	const size_t &sampleCount = channelData.size();

	std::fseek(fp_, long(headerSize_ + sampleStartIdx * getBlockAlign()), SEEK_SET);
#if 1
	T sample;
	size_t samplesRead;
	for (size_t idx = 0; idx < sampleCount; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
		{
			samplesRead = std::fread(&sample, getBytesPerSample(), 1, fp_);
			//samplesRead = std::fread(&sample, sizeof(T), 1, fp_);
			if (channel == ch)
			{
				if (1 == samplesRead)
					channelData[idx] = sample;
				else return idx;
			}
		}

	return sampleCount;
#elif 0
	if (channel > 0)
		std::fseek(fp_, long(headerSize_ + sampleStartIdx * getBlockAlign() + channel * getBytesPerSample()), SEEK_SET);

	T sample;
	size_t samplesRead;
	for (size_t idx = 0; idx < sampleCount; ++idx)
	{
		samplesRead = std::fread(&sample, getBytesPerSample(), 1, fp_);
		//const size_t samplesRead = std::fread(&sample, sizeof(T), 1, fp_);
		if (1 == samplesRead)
			channelData[idx] = sample;
		else return idx;

		// Read the other channels.
		samplesRead = std::fread(&sample, getBytesPerSample(), getNumberOfChannels() - 1, fp_);
		// TODO [check] >> Is it correct to return idx?
		if (0 == samplesRead || getNumberOfChannels() - 1 != samplesRead) return idx;
	}

	return sampleCount;
#elif 0
	std::vector<T> sample(getNumberOfChannels());
	size_t samplesRead;
	for (size_t idx = 0; idx < sampleCount; ++idx)
	{
		samplesRead = std::fread(&sample[0], getBytesPerSample(), getNumberOfChannels(), fp_);
		//samplesRead = std::fread(&sample[0], sizeof(T), getNumberOfChannels(), fp_);
		if (getNumberOfChannels() == samplesRead)
			channelData[idx] = sample[channel];
		else return idx;
	}

	return sampleCount;
#else
	std::vector<T> sample(getNumberOfChannels() * sampleCount);
	const size_t samplesRead = std::fread(&sample[0], getBytesPerSample(), getNumberOfChannels() * sampleCount, fp_) / getNumberOfChannels();
	//const size_t samplesRead = std::fread(&sample[0], sizeof(T), getNumberOfChannels() * sampleCount, fp_);
	for (size_t idx = 0; idx < samplesRead; ++idx)
		channelData[idx] = sample[idx * getNumberOfChannels() + channel];

	return samplesRead;
#endif
}

}  // namespace swl


#endif  // __SWL_UTIL__WAVE_DATA__H_
