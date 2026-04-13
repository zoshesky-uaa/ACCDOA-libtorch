#include "audio/audio.h"

public void feature_extract(AudioDevice& audioDevice) {
	float* tempBuffer = new float[audioDevice.framelimit * audioDevice.channels];
	bool r = audioDevice.read(tempBuffer);
	while (!r) {
		r = audioDevice.read(tempBuffer);
	}
}
	