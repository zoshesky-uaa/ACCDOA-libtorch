#pragma once
#include "../include/config.h"
#include "../include/features.h"
#include "../include/audio.h"
#include "../include/fsm.h"

// xtensor
#include <xtensor/containers/xarray.hpp>

// torch
#include <torch/torch.h>

struct GenerateCmd {
	std::string device_name = "";
	std::string zarr_path = ""; // Direct path to a trial zarr, e.g., "data/trial_1.zarr"
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(GenerateCmd, device_name, zarr_path)
};

class Generate {
private:
	SystemConfig& config;
	// Shared buffer for active features
	xt::xtensor<float, 2> sed_features;
	xt::xtensor<float, 2> doa_features;

	// Process classes
	AudioDevice audio_device;
	FeatureExtractor feature_extractor;
	Writer writer;

	int tick = 0;
public:
	Generate(const GenerateCmd& cmd,
		SystemConfig& config) :
		sed_features({ 1, config.mel_bins }, 0.0f),
		doa_features({ 5, config.mel_bins }, 0.0f),
		audio_device(cmd.device_name, config),
		feature_extractor(config, sed_features, doa_features),
		writer(cmd.zarr_path, config),
		config(config)
	{
		audio_device.start();
		std::cout << "Intiated: Writing to " << cmd.zarr_path << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1)); 
		std::cout << "START" << std::endl;
		while (config.on.load(std::memory_order_relaxed)) {
			if (feature_extractor.feature_extract(audio_device)) {
				writer.add_frame(sed_features, doa_features);
				if (writer.count >= config.frame_max) {
					return;
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

	~Generate() {
		config.on.store(false, std::memory_order_relaxed);
	}
};