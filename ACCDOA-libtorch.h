#pragma once
#include "features.h"
#include "audio/audio.h"

struct SystemConfig {
	size_t sample_rate;
	size_t channels;
	size_t fft_size;
	size_t hop_length;
	size_t fft_bins;
	size_t mel_bins;
};

class ACCDOA {
	private:
		bool training;
		AudioDevice audio_device;
		FeatureExtractor feature_extractor;
		
	public:
		ACCDOA(bool training_mode, 
				char* device_name,
				SystemConfig config) 
			: training(training_mode), 
			audio_device(DeviceType::Capture, device_name, config),
			feature_extractor(config) {
			if (training) {
				audio_device.start();
				// Setup a while loop that does extract on loop attached to an atomic booleans 
			}
		}
};