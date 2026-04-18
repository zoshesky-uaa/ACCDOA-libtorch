#pragma once
#include "audio/features.h"
#include "audio/audio.h"
#include "audio/fsm.h"
#include <nlohmann/json.hpp>


struct SystemConfig {
	size_t sample_rate; // Sample rate for audio capture (e.g., 16000 Hz)
	size_t channels; // Number of audio channels (e.g., 4 for first-order ambisonics)
	size_t fft_size; // FFT size for the STFT
	size_t hop_length; // Hop length for the STFT, typically fft_size / 2
	size_t fft_bins; // Number of FFT bins (fft_size / 2 + 1)
	size_t mel_bins; // Number of Mel bands for the log-mel spectrogram
	size_t feature_dim; // Total feature dimension (mel_bins + 2 * fft_bins)
	size_t frame_max; // Simulation maximum length in frames
	std::atomic<bool> on;
};

class ACCDOA {
	private:
		bool training;
		// Shared buffer for active features
		std::vector<float> features;

		// Process classes
		AudioDevice audio_device;
		FeatureExtractor feature_extractor;
		Writer writer;
		
	public:
		ACCDOA(const bool& training_mode, 
				const std::string& device_name,
				const std::string& zarr_path,	
				SystemConfig& config) 
			: training(training_mode), 
			features(config.feature_dim, 0.0f),
			audio_device(device_name, config),
			feature_extractor(config, std::ref(features)),
			writer(zarr_path, config)
			{
			if (training) {
				audio_device.start();
				while (config.on) {
					if (feature_extractor.feature_extract(audio_device)) {
						writer.addFrame(features);
						if (writer.count >= config.frame_max) {
							config.on.store(false);
						}
					}
				}
			}
		}
};