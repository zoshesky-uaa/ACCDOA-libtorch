#include "train.h"

int main() {
	std::cout << std::unitbuf;
	return model_process<Train, TrainCmd>();
}