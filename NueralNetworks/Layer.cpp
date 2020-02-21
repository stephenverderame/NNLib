#include "Layer.h"

Layer::Layer() : previous(nullptr)
{
}


Layer::~Layer()
{
}

void Layer::setPreviousLayer(Layer * p)
{
	previous = p;
}

ConvLayer::ConvLayer(size_t size, size_t depth, size_t stride, size_t padding) : stride(stride), padding(padding)
{
	biases.resize(depth, 0);
	for (size_t i = 0; i < depth; ++i) {
		kernels.emplace_back(size, size);
		randomize(kernels[i]);
	}
	randomize(biases);
}

std::vector<Matrix<>> ConvLayer::calculate(std::vector<Matrix<>>& inputs)
{
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < kernels.size(); ++i) {
		out.push_back(kernels[i].applyAsKernel(inputs[i % inputs.size()].zeroPad(padding), stride));
		out[i] += biases[i];
	}
	return out;
}

std::vector<Matrix<>> ConvLayer::backprop(std::vector<Matrix<>>& costs)
{
	return std::vector<Matrix<>>();
}

std::vector<Matrix<>> ConvLayer::getWeights()
{
	return kernels;
}

ActivationLayer::ActivationLayer(Activation f, Activation fP) : function(f), functionPrime(fP)
{
	
}

std::vector<Matrix<>> ActivationLayer::calculate(std::vector<Matrix<>>& inputs)
{
	temp = inputs;
	std::vector<Matrix<>> out;
	for (Matrix<> & m : inputs) {
		out.push_back(function(m));
	}
	return out;
}

std::vector<Matrix<>> ActivationLayer::backprop(std::vector<Matrix<>>& costs)
{
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < costs.size(); ++i) {
		out.push_back(hadamard(costs[i], functionPrime(temp[i])));
	}
	return out;
}

std::vector<Matrix<>> ActivationLayer::getWeights()
{
	return std::vector<Matrix<>>();
}

PoolingLayer::PoolingLayer(size_t size, size_t stride) : size(size), stride(stride)
{
}

std::vector<Matrix<>> PoolingLayer::calculate(std::vector<Matrix<>>& inputs)
{
	std::vector<Matrix<>> out;
	for (Matrix<> & m : inputs)
		out.push_back(m.maxPool(size, stride));
	return out;
}

std::vector<Matrix<>> PoolingLayer::backprop(std::vector<Matrix<>>& costs)
{
	return std::vector<Matrix<>>();
}

std::vector<Matrix<>> PoolingLayer::getWeights()
{
	return std::vector<Matrix<>>();
}

FCLayer::FCLayer(size_t inputSize, size_t outputSize) : inputSize(inputSize), outputSize(outputSize)
{
	weights.resize(outputSize, inputSize);
	biases.resize(outputSize);
	randomize(weights);
	randomize(biases);
}

std::vector<Matrix<>> FCLayer::calculate(std::vector<Matrix<>>& inputs)
{
	Vector<> t;
	for (auto mat : inputs) {
		auto vector = static_cast<Vector<>>(mat);
		t.insert(t.end(), vector.begin(), vector.end());
	}
	temp.resize(1);
	temp[0] = t;
	std::vector<Matrix<>> out;
	out.push_back(static_cast<Matrix<>>(weights * t + biases));
	return out;
}

std::vector<Matrix<>> FCLayer::backprop(std::vector<Matrix<>>& costs)
{
	assert(costs.size() == 1 && "Backprop on fully connected layer is expected to be a vector");
	Vector<> dCdB = static_cast<Vector<>>(costs[0]);
	Matrix<> dCdW = static_cast<Matrix<>>(dCdB) * transpose(temp[0]);
	biases -= learningRate * dCdB;
	weights -= learningRate * dCdW;

	std::vector<Matrix<>> out;
	out.push_back(transpose(previous->getWeights()[0]) * static_cast<Matrix<>>(dCdB)); //normally multiply by previous layer weights
	return out;
}

std::vector<Matrix<>> FCLayer::getWeights()
{
	return std::vector<Matrix<>>(1, weights);
}
