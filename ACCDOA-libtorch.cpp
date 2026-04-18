// ACCDOA-libtorch.cpp : Defines the entry point for the application.

#include <torch/torch.h>
#include "audio/audio.h"	
#include "ACCDOA-libtorch.h"
#include <atomic>

using namespace std;
struct InputCommand {	
	string device_name;
	string zarr_path;
	bool training_mode;
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(InputCommand, device_name, zarr_path, training_mode)
};


static constexpr SystemConfig config{
	.sample_rate = 16000,
	.channels = 4, // Not designed to be changed, harcoded for now. .
	.fft_size = 1024,
	.hop_length = 1024/2,
	.fft_bins = (1024/2) + 1,
	.mel_bins = 64,
	.feature_dim = 64 + (2 * ((1024 / 2) + 1)),
	.frame_max = 20000, // 650 seconds at 32.5 millseconds per tick (sample rate / (hop_length = framelimit))
	.on = true
};

int main() {
	// Setup stdin to read commands (i.e. device name, start)
	// Setup another thread for stdout to print ticks
	cout << "Provide JSON signature:" << endl;
	try {
		nlohmann::json incoming_json;
		cin >> incoming_json;
		InputCommand input_command = incoming_json.get<InputCommand>();
		ACCDOA accdoa(ref(input_command.training_mode), ref(input_command.device_name), ref(input_command.zarr_path), ref(config));
		while (config.on) {

		}
	} catch (const nlohmann::json::exception& e) {
		cerr << "JSON Parsing Error: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}
