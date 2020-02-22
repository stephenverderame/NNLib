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

ConvLayer::ConvLayer(size_t size, size_t numKernels, size_t depth, size_t stride, size_t padding, bool paramSharing) : stride(stride), padding(padding), parameterSharing(paramSharing)
{
	if (padding == -1 && stride == 1) padding = (size - 1) / 2; //assuming a stride of 1, this is the zero padding needed to prevent downsizing
	biases.resize(numKernels, 0);
	for (size_t i = 0; i < numKernels; ++i) {
		std::vector<Matrix<>> kernel;
		for (size_t j = 0; j < depth; ++j) {
			kernel.emplace_back(size, size);
			randomize(kernel[j]);
		}
		kernels.push_back(kernel);
	}
	randomize(biases);
}

std::vector<Matrix<>> ConvLayer::calculate(std::vector<Matrix<>>& inputs)
{
	/**
	* Each kenel has an equal depth to the input depth. If parameter sharing is enabled, every kernel has the same weights for all of its depth
	*/
	assert(kernels[0].size() == inputs.size() && "Depth between conv layer and input doesn't match");
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < kernels.size(); ++i) {
		Matrix<> temp((inputs[0].rows() - kernels[0][0].rows() + 2 * padding) / stride + 1, (inputs[0].rows() - kernels[0][0].rows() + 2 * padding) / stride + 1);
		for (size_t j = 0; j < inputs.size(); ++j) {
			size_t kernelDepth = parameterSharing ? 0 : j;
			temp += kernels[i][kernelDepth].applyAsKernel(inputs[j].zeroPad(padding), stride);
		}
		temp += biases[i];
		out.push_back(temp);
	}
	return out;
}

std::vector<Matrix<>> ConvLayer::backprop(std::vector<Matrix<>>& costs)
{
	return std::vector<Matrix<>>();
}

std::vector<Matrix<>> ConvLayer::getWeights()
{
	assert("Not implemented yet");
	return kernels[0];
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
