#pragma once
#include "Matrix.h"
namespace mnist {
	std::vector<Vector<>> readImages(const char * filename);
	std::vector<uint8_t> readLabels(const char * filename);
	int reverseInt(int i);
}