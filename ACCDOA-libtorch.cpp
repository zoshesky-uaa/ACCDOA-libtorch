// ACCDOA-libtorch.cpp : Defines the entry point for the application.
#include <torch/torch.h>
#include <atomic>
#include "ACCDOA-libtorch.h"

using namespace std;
struct InputCommand {	
	string device_name = "";
	string zarr_path = "";
	bool training_mode = false;
	NLOHMANN_DEFINE_TYPE_INTRUSIVE(InputCommand, device_name, zarr_path, training_mode)
};

static constexpr int DEBUG_LIMIT = 26;
std::optional<InputCommand> read_input(SystemConfig& config, bool JSON) {
	std::string raw_input;
	int i = 1;
	cout << "Provide JSON signature:" << endl;
	while (i < DEBUG_LIMIT) {
		if (!std::getline(std::cin, raw_input) || raw_input.empty()) continue;
		if (raw_input == "exit") {
			break;
		}
		if (nlohmann::json::accept(raw_input) && JSON) {
			try {
				auto j = nlohmann::json::parse(raw_input);
				return j.get<InputCommand>();
			}
			catch (const std::exception& e) {
				std::cerr << "Schema Error: " << e.what() << std::endl;
			}
		}
		else {
			std::cerr << "Invalid JSON or command, try again." << std::endl;
		}
		std::cout << "Provide JSON signature (Attempt " << ++i << "):" << std::endl;
	}
	config.on.store(false);
	std::cout << "Stopping application." << std::endl;
	return nullopt;
}

std::thread processor_thread(const InputCommand& cmd, SystemConfig& config) {
	return std::thread([cmd, &config]() {
		try {
			ACCDOA accdoa(cmd.training_mode, cmd.device_name, cmd.zarr_path, config);
		}
		catch (const std::exception& e) {
			std::cerr << "Error during processing: " << e.what() << std::endl;
			config.on.store(false);
		}
	});
}

std::thread read_input_thread(SystemConfig& config) {
	return std::thread([&config]() {
		read_input(config, false);
	});
}

int main() {
	SystemConfig config = SystemConfig();
	auto input_command = read_input(config, true);
	if (input_command.has_value()) {
		std::thread thread_1 = processor_thread(input_command.value(), config);
		std::thread thread_2 = read_input_thread(config);
		while (config.on) {
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
		if (thread_1.joinable()) thread_1.join();
		if (thread_2.joinable()) thread_2.join();
	}
	return 0;
}