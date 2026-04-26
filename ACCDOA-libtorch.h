#pragma once
#include "config.h"
#include "audio/features.h"
#include "audio/audio.h"
#include "audio/fsm.h"
#include <nlohmann/json.hpp>

class ACCDOA {
	private:
		bool training;
		// Shared buffer for active features
		std::vector<float> features;

		// Process classes
		AudioDevice audio_device;
		FeatureExtractor feature_extractor;
		Writer writer;
		SystemConfig& config;
		int tick = 0;
	public:
		ACCDOA(const bool training_mode, 
				const std::string device_name,
				const std::string zarr_path,	
				SystemConfig& config) 
			: training(training_mode), 
			features(config.feature_dim, 0.0f),
			audio_device(device_name, config),
			feature_extractor(config, features),
			writer(zarr_path, config, training_mode),
			config(config)
			{
			std::cout << "Initialized ACCDOA with device: " << device_name << " and zarr path: " << zarr_path << std::endl;
			if (training) {
				audio_device.start();
				std::cout << "START" << std::endl;
				while (config.on) {
					if (feature_extractor.feature_extract(audio_device)) {
						writer.addFrame(features);
						if (writer.count >= config.frame_max) {
							config.on.store(false);
						}
						std::cout << std::to_string(tick++) << std::endl;
					}
				}
				if (writer.count != config.frame_max) {
					std::cerr << "Warning: Early termination detected. Potential data loss, delete associated files." << std::endl;
					return;
				}
				std::cout << "END" << std::endl;

			}
		}
		~ACCDOA() {
			config.on.store(false);
		}
};

