#pragma once
#include "features.h"
#include "audio/audio.h"

class ACCDOA {
	private:
		bool training;
		FeatureExtractor feature_extractor;
		AudioDevice audio_device;

};