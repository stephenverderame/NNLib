#pragma once
#include "Matrix.h"
namespace mnist {
	/**
	* Functions to read the images from the MNIST database.
	* Both functions should return a vector of equal size where label[i] is the number represented by image[i]
	* Images are size 28 x 28 
	*/
	std::vector<Vector<>> readImages(const char * filename);
	std::vector<uint8_t> readLabels(const char * filename);

	/**Helper function for reading the MNIST database*/
	int reverseInt(int i);
}