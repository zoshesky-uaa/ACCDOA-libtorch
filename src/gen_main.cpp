#include "generate.h"

int main() {
	std::cout << std::unitbuf;
	return model_process<Generate, GenerateCmd>();
}

