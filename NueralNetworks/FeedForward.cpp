#define _CRT_SECURE_NO_WARNINGS
#include "FeedForward.h"
#include <stdarg.h>
#include <functional>
#include <thread>
#ifdef USE_GPGPU
#include <CL\cl.h>
#endif

void FeedForward::initMatrices()
{
	activations.resize(layers.size());
	z.resize(layers.size());
	for (size_t i = 1; i < layers.size(); ++i) {
		biases.emplace_back(layers[i]);
		randomize(biases[i - 1]);
		weights.emplace_back(layers[i], layers[i - 1]);
		randomize(weights[i - 1]);
	}
}

void FeedForward::randomize(Matrix<> & mat)
{
	for (size_t i = 0; i < mat.size(); ++i)
		mat[i] = (double)rand() / RAND_MAX;
}

void FeedForward::randomize(Vector<>& vec)
{
	for (size_t i = 0; i < vec.size(); ++i)
		vec[i] = (double)rand() / RAND_MAX;
}

FeedForward::FeedForward(std::initializer_list<size_t> layers)
{
	this->layers.insert(this->layers.end(), layers.begin(), layers.end());
	initMatrices();
}

FeedForward::FeedForward(size_t numLayers, size_t * layers)
{
	this->layers.resize(numLayers);
	for (size_t i = 0; i < numLayers; ++i)
		this->layers[i] = layers[i];
	initMatrices();
}

FeedForward::FeedForward(size_t numLayers, ...)
{
	layers.resize(numLayers);
	va_list args;
	va_start(args, numLayers);
	for (size_t i = 0; i < numLayers; ++i)
		layers[i] = va_arg(args, size_t);
	va_end(args);
	initMatrices();
}

void FeedForward::resize(std::vector<size_t>& layers)
{
	this->layers = layers;
	weights.clear();
	biases.clear();
	initMatrices();
}

Vector<> FeedForward::calculate(const Vector<> & input) const
{
	activations[0] = input;
	for (size_t i = 1; i < activations.size(); ++i) {
		z[i] = weights[i - 1] * activations[i - 1] + biases[i - 1];
		activations[i] = f(z[i]);
	}
	return activations[activations.size() - 1];
}

void FeedForward::backprop(const Vector<> & out, const Vector<> & real)
{
	Vector<> dCdA, dCdB;
	Matrix<> dCdW;
	dCdA = 2.0 * (out - real); //dCost
	for (size_t i = activations.size() - 1; i > 0; --i) {
		dCdB = hadamard(dCdA, fPrime(z[i]));
		dCdW = (Matrix<>)dCdB * transpose(activations[i - 1]);
		if(i > 1)
			dCdA = transpose(weights[i - 1]) * dCdB;

		if (learningRates.empty()) {
			weights[i - 1] -= learningRate * dCdW;
			biases[i - 1] -= learningRate * dCdB;
		}
		else {
			weights[i - 1] -= learningRates[i - 1] * dCdW;
			biases[i - 1] -= learningRates[i - 1] * dCdB;
		}

	}
}

void FeedForward::setLearningRates(double r0, ...)
{
	assert(layers.size() > 2 && "Only pass numLayers - 1 learningRates");
	va_list args;
	va_start(args, r0);
	learningRates.push_back(r0);
	for (size_t i = 0; i < layers.size() - 2; ++i) {
		learningRates.push_back(va_arg(args, double));
	}
	va_end(args);
}

double vectorSum(Vector<> & v)
{
	double sum = 0;
	size_t index = 0;
	switch (v.size() % 4) {
	case 0: do {
			sum += v[index++];
	case 3:	sum += v[index++];
	case 2:	sum += v[index++];
	case 1:	sum += v[index++];
		} while (index < v.size());
	}
	return sum;
}

double vectorSum(Vector<> && v)
{
	double sum = 0;
	size_t index = 0;
	switch (v.size() % 4) {
	case 0: do {
		sum += v[index++];
	case 3:	sum += v[index++];
	case 2:	sum += v[index++];
	case 1:	sum += v[index++];
		} while (index < v.size());
	}
	return sum;
}

std::unique_ptr<NeuralNetwork> NetworkSerializer::loadFF(FILE * f) {
	size_t layerNum;
	fread(&layerNum, sizeof(size_t), 1, f);
	auto layers = std::make_unique<size_t[]>(layerNum);
	fread(layers.get(), sizeof(size_t), layerNum, f);
	auto net = std::make_unique<FeedForward>(layerNum, layers.get());
	for (size_t i = 0; i < layerNum - 1; ++i) {
		auto w = std::make_unique<double[]>(net->weights[i].size());
		auto b = std::make_unique<double[]>(net->biases[i].size());
		fread(w.get(), sizeof(double), net->weights[i].size(), f);
		fread(b.get(), sizeof(double), net->biases[i].size(), f);
		std::vector<double> wv, bv;
		wv.insert(wv.end(), w.get(), w.get() + net->weights[i].size());
		bv.insert(bv.end(), b.get(), b.get() + net->biases[i].size());
		net->weights[i].loadFromRawData(wv);
		net->biases[i] = bv;
	}
	return net;
}
void NetworkSerializer::saveFF(FILE * f, NeuralNetwork * nn) {
	FeedForward * ff = (FeedForward*)nn;
	size_t s = ff->layers.size();
	fwrite(&s, sizeof(s), 1, f);
	fwrite(ff->layers.data(), sizeof(size_t), s, f);
	for (int i = 0; i < s - 1; ++i) {
		auto data = ff->weights[i].getData();
		fwrite(data.data(), sizeof(double), data.size(), f);
		auto bdata = (std::vector<double>)ff->biases[i];
		fwrite(bdata.data(), sizeof(double), bdata.size(), f);
	}
}
void FeedForward::threadHelper(FeedForward & nn, const std::vector<Vector<>>& inputs, std::function<Vector<>(const Vector<>&x, size_t id)>& calcReal, size_t iterations)
{
	for (size_t i = 0; i < iterations; ++i) {
		const Vector<> & x = inputs[i % inputs.size()];
		Vector<> r = calcReal(x, i % inputs.size());
		Vector<> y = nn.calculate(x);
		nn.backprop(y, r);
	}
}
void FeedForward::trainMultithreaded(const std::vector<Vector<>>& inputs, std::function<Vector<>(const Vector<>&x, size_t id)> calcReal, uint8_t threads, size_t batchSize, size_t totalIterations)
{
	printf("Training: 0.00%%\r");
	std::vector<FeedForward> buffers(threads, *this);
	for (size_t i = 0; i < totalIterations / batchSize / threads; ++i) {
		std::vector<std::thread> threadList;
		for (uint8_t j = 0; j < threads; ++j)
			threadList.push_back(std::thread(&FeedForward::threadHelper, std::ref(buffers[j]), std::cref(inputs), std::ref(calcReal), batchSize));
		for (uint8_t j = 0; j < threads; ++j)
			threadList[j].join();
		auto newNetwork = buffers[0];
		for (uint8_t j = 1; j < buffers.size(); ++j) {
			for (uint8_t k = 0; k < newNetwork.biases.size(); ++k)
				newNetwork.biases[k] += buffers[j].biases[k];
			for (uint8_t k = 0; k < newNetwork.weights.size(); ++k)
				newNetwork.weights[k] += buffers[j].weights[k];
		}
		for (uint8_t k = 0; k < newNetwork.biases.size(); ++k)
			newNetwork.biases[k] *= 1.0 / threads;
		for (uint8_t k = 0; k < newNetwork.weights.size(); ++k)
			newNetwork.weights[k] *= 1.0 / threads;
		std::fill(buffers.begin(), buffers.end(), newNetwork);
		printf("Training: %.2f%%\r", (double)i * batchSize * threads / totalIterations * 100.0);
	}
	printf("Training: 100.00%%\n");
}
#ifdef _GPGPU
void FeedForward::trainGPU(const std::vector<Vector<>>& inputs, const std::vector<Vector<>>& outputs, size_t batchSize, size_t totalIterations)
{
	
}
#endif
