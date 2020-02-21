#define _CRT_SECURE_NO_WARNINGS
#include "NeuralNetwork.h"

std::unique_ptr<NeuralNetwork> NetworkSerializer::loadNet(const char * filename)
{
	FILE * f = fopen(filename, "rb");
	if (f != NULL) {
		uint16_t type;
		fread(&type, sizeof(short), 1, f);
		std::unique_ptr<NeuralNetwork> nn;
		switch (type) {
		case nt_feedForward:
			nn = loadFF(f);
			break;
		}
		fclose(f);
		return nn;
	}
	throw std::string("Invalid file");
}

void NetworkSerializer::saveNet(NeuralNetwork * nn, NetworkTypes type, const char * filename)
{
	FILE * f = fopen(filename, "wb");
	fwrite(&type, sizeof(uint16_t), 1, f);
	switch (type) {
	case nt_feedForward:
		saveFF(f, nn);
		break;
	}
	fclose(f);
}
