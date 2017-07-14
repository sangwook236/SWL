#if !defined(__SWL_UTIL__WAVE_DATA__H_ )
#define __SWL_UTIL__WAVE_DATA__H_ 1


#include "swl/util/ExportUtil.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cassert>


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

	template<typename SampleType, typename DataType>
	size_t readAllChannelData(const size_t startSampleIdx, std::vector<std::vector<DataType> > &allChannelData) const;
	template<typename SampleType, typename DataType>
	size_t readChannelData(const size_t channel, const size_t startSampleIdx, std::vector<DataType> &channelData) const;

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
	size_t getFileSize() const;  // [byte].
	double getFileLength() const  // [sec].
	{
		return (double)getDataSize() / (double)getBytesPerSecond();
	}

private:
	FILE *fp_;
	bool isOpened_;

	WaveHeader header_;
	size_t headerSize_;
};

template<typename SampleType, typename DataType>
size_t WaveData::readAllChannelData(const size_t startSampleIdx, std::vector<std::vector<DataType> > &allChannelData) const
{
	if (!isOpened() || allChannelData.empty() || allChannelData.size() != getNumberOfChannels()) return 0;
	const size_t sampleSize = sizeof(SampleType);
	assert(getBytesPerSample() == sampleSize);

	const size_t &numSamples = allChannelData[0].size();
	for (size_t ch = 1; ch < getNumberOfChannels(); ++ch)
		if (allChannelData[ch].size() != numSamples) return 0;

	std::fseek(fp_, long(headerSize_ + startSampleIdx * getBlockAlign()), SEEK_SET);
#if 1
	SampleType sample;
	size_t numSamplesRead;
	for (size_t idx = 0; idx < numSamples; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
		{
			numSamplesRead = std::fread(&sample, sampleSize, 1, fp_);
			if (1 == numSamplesRead)
				allChannelData[ch][idx] = sample;
			else return idx;
		}

	return numSamples;
#elif 0
	std::vector<SampleType> sample(getNumberOfChannels());
	size_t numSamplesRead;
	for (size_t idx = 0; idx < numSamples; ++idx)
	{
		numSamplesRead = std::fread(&sample[0], sampleSize, getNumberOfChannels(), fp_);
		if (getNumberOfChannels() == numSamplesRead)
			for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
				allChannelData[ch][idx] = sample[ch];
		else return idx;
	}

	return numSamples;
#else
	std::vector<SampleType> sample(getNumberOfChannels() * numSamples);
	const size_t numSamplesRead = std::fread(&sample[0], sampleSize, getNumberOfChannels() * numSamples, fp_);
	for (size_t idx = 0; idx < numSamplesRead; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
			allChannelData[ch][idx] = sample[idx * getNumberOfChannels() + ch];

	return numSamplesRead;
#endif
}

template<typename SampleType, typename DataType>
size_t WaveData::readChannelData(const size_t channel, const size_t startSampleIdx, std::vector<DataType> &channelData) const
{
	if (!isOpened() || channel >= getNumberOfChannels()) return 0;
	const size_t sampleSize = sizeof(SampleType);
	assert(getBytesPerSample() == sampleSize);

	const size_t &numSamples = channelData.size();

	std::fseek(fp_, long(headerSize_ + startSampleIdx * getBlockAlign()), SEEK_SET);
#if 1
	SampleType sample;
	size_t numSamplesRead;
	for (size_t idx = 0; idx < numSamples; ++idx)
		for (size_t ch = 0; ch < getNumberOfChannels(); ++ch)
		{
			numSamplesRead = std::fread(&sample, sampleSize, 1, fp_);
			if (channel == ch)
			{
				if (1 == numSamplesRead)
					channelData[idx] = sample;
				else return idx;
			}
		}

	return numSamples;
#elif 0
	if (channel > 0)
		std::fseek(fp_, long(headerSize_ + startSampleIdx * getBlockAlign() + channel * sampleSize), SEEK_SET);

	SampleType sample;
	size_t numSamplesRead;
	for (size_t idx = 0; idx < numSamples; ++idx)
	{
		const size_t numSamplesRead = std::fread(&sample, sampleSize, 1, fp_);
		if (1 == numSamplesRead)
			channelData[idx] = sample;
		else return idx;

		// Read the other channels.
		numSamplesRead = std::fread(&sample, sampleSize, getNumberOfChannels() - 1, fp_);
		// TODO [check] >> Is it correct to return idx?
		if (0 == numSamplesRead || getNumberOfChannels() - 1 != numSamplesRead) return idx;
	}

	return numSamples;
#elif 0
	std::vector<SampleType> sample(getNumberOfChannels());
	size_t numSamplesRead;
	for (size_t idx = 0; idx < numSamples; ++idx)
	{
		numSamplesRead = std::fread(&sample[0], sampleSize, getNumberOfChannels(), fp_);
		if (getNumberOfChannels() == numSamplesRead)
			channelData[idx] = sample[channel];
		else return idx;
	}

	return numSamples;
#else
	std::vector<SampleType> sample(getNumberOfChannels() * numSamples);
	//const size_t numSamplesRead = std::fread(&sample[0], sampleSize, getNumberOfChannels() * numSamples, fp_);
	for (size_t idx = 0; idx < numSamplesRead; ++idx)
		channelData[idx] = sample[idx * getNumberOfChannels() + channel];

	return numSamplesRead;
#endif
}

}  // namespace swl


#endif  // __SWL_UTIL__WAVE_DATA__H_
