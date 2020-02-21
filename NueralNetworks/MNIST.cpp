#include "MNIST.h"
#include <fstream>

std::vector<Vector<>> mnist::readImages(const char * filename)
{
	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open() && "Cannot open file!");
	int magicNum = 0, rows = 0, cols = 0, numImages = 0;
	file.read((char*)&magicNum, sizeof(int));
	magicNum = reverseInt(magicNum);
	assert(magicNum == 2051 && "Invalid MNIST file!");
	file.read((char*)&numImages, sizeof(int));
	numImages = reverseInt(numImages);
	file.read((char*)&rows, sizeof(int));
	rows = reverseInt(rows);
	file.read((char*)&cols, sizeof(int));
	cols = reverseInt(cols);

	std::vector<Vector<>> images;
	for (int i = 0; i < numImages; ++i) {
		Vector<> img(rows * cols);
		char * b = new char[rows * cols];
		file.read(b, rows * cols);
		for (int j = 0; j < rows * cols; ++j)
			img[j] = b[j];
		delete[] b;
		images.push_back(img);
	}
	return images;

}

std::vector<uint8_t> mnist::readLabels(const char * filename)
{
	std::ifstream file(filename, std::ios::binary);
	assert(file.is_open() && "Cannot open file!");
	int magicNum = 0, labelNum = 0;
	file.read((char*)&magicNum, sizeof(int));
	magicNum = reverseInt(magicNum);
	assert(magicNum == 2049 && "Invalid MNIST label!");
	file.read((char*)&labelNum, sizeof(int));
	labelNum = reverseInt(labelNum);
	
	char * labels = new char[labelNum];
	file.read(labels, labelNum);

	std::vector<uint8_t> labelVector;
	for (int i = 0; i < labelNum; ++i)
		labelVector.push_back(labels[i]);
	delete[] labels;
	return labelVector;
}

int mnist::reverseInt(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
