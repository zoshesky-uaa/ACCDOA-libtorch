#include "inference.h"

int main() {
	std::cout << std::unitbuf;
	return model_process<Inference, InferenceCmd>();
}