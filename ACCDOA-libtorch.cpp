// ACCDOA-libtorch.cpp : Defines the entry point for the application.

#include <torch/torch.h>
#include "audio/audio.h"	
#include "ACCDOA-libtorch.h"

using namespace std;

static constexpr SystemConfig config{
	.sample_rate = 16000,
	.channels = 4,
	.fft_size = 1024,
	.hop_length = 512,
	.fft_bins = 513,
	.mel_bins = 64
};

int main() {
	// Setup stdin to read commands (i.e. device name, start)
	// Setup another thread for stdout to print ticks
	cout << "Hello CMake." << endl;
	return 0;
}
