#include "Layer.h"

std::vector<Matrix<>> Layer::unflatten(const Matrix<>& vector, const dimensionData & dims)
{
	if(vector.cols() != 1 && vector.size() != dims.width * dims.height * dims.depth) throw UndefinedException(ERR_STR("Input vector cannot be correctly unflattened to specified with and depth"));
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < dims.depth; ++i)
		out.emplace_back(dims.height, dims.width);
	for (size_t i = 0; i < vector.size(); ++i) {
		size_t depth = i / (dims.width * dims.height);
		size_t element = i % (dims.width * dims.height);
		out[depth][element] = vector.get(i);
	}
	return out;

}

Layer::Layer() : previous(nullptr), learningRate(1.0)
{
}


Layer::~Layer()
{
}

void Layer::setPreviousLayer(Layer * p)
{
	previous = p;
}

dimensionData Layer::getOutputDimensions()
{
	return outputDimensions;
}

void Layer::setLearningRate(double r)
{
	learningRate = r;
}

double Layer::getLearningRate()
{
	return learningRate;
}

dimensionData Layer::connectLayer(dimensionData input)
{
	inputDimensions = input;
	outputDimensions = connectLayerVirtual();
	return outputDimensions;
}

ConvLayer::ConvLayer(size_t size, size_t numKernels, size_t stride, size_t padding, bool paramSharing) : stride(stride), padding(padding), parameterSharing(paramSharing)
{
	if (padding == defaultVal && stride == 1) this->padding = (size - 1) / 2; //assuming a stride of 1, this is the zero padding needed to prevent downsizing
	kernelSize = size;
	biases.resize(numKernels, 0);
	randomize(biases);
}

std::vector<Matrix<>> ConvLayer::calculate(std::vector<Matrix<>>& inputs)
{
	/**
	* Each kenel has an equal depth to the input depth. If parameter sharing is enabled, every kernel has the same weights for all of its depth
	*/
	for (auto& i : inputs)
		temp.push_back(i.zeroPad(padding));
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < kernels.size(); ++i) {
		Matrix<> buf((inputs[0].rows() - kernels[0][0].rows() + 2 * padding) / stride + 1, (inputs[0].rows() - kernels[0][0].rows() + 2 * padding) / stride + 1);
		for (size_t j = 0; j < inputs.size(); ++j) {
			size_t kernelDepth = parameterSharing ? 0 : j;
			//temp is the zeropadded inputs
			buf += kernels[i][kernelDepth].applyAsKernel(temp[j], stride);
		}
		buf += biases[i];
		out.push_back(buf);
	}
//	outputDimensions = { out[0].cols(), out[0].rows(), out.size() };
	return out;
}

std::vector<Matrix<>> ConvLayer::backprop(std::vector<Matrix<>>& costs)
{
	//dK[x][y] is sum of every gradient times every corresponding input that that specific weight had an impact on
	assert(parameterSharing && "Only implemented for param sharing right now");
	//convolve the gradient with the layer input
	std::vector<Matrix<>> output;
	for (size_t i = 0; i < costs.size(); ++i) { //dimensions of output = dimensions of gradient therefore depth of output = num kernels = depth of gradient
		Matrix<> buf(kernels[0][0].rows(), kernels[0][0].cols());
		for (size_t j = 0; j < temp.size(); ++j) {
			//temp is the zero padded inputs
			buf += costs[i].applyAsKernel(temp[j], stride);
		}
		Matrix<> oBuf = fullKernelConvolution(kernels[i][0].rotate180(), costs[i]);
		kernels[i][0] -= learningRate * buf;
		double costSum = 0;
		for (size_t j = 0; j < costs[i].size(); ++j)
			costSum += costs[i][j];
		biases[i] -= learningRate * costSum;		
		output.push_back(removePadding(oBuf, padding));
	}
	return output;
}

dimensionData ConvLayer::connectLayerVirtual()
{
	for (size_t i = 0; i < kernelSize; ++i) {
		std::vector<Matrix<>> kernel;
		if (!parameterSharing) {
			for (size_t j = 0; j < inputDimensions.depth; ++j) {
				kernel.emplace_back(kernelSize, kernelSize);
				randomize(kernel[j]);
			}
		}
		else {
			kernel.emplace_back(kernelSize, kernelSize);
			randomize(kernel[0]);
		}
		kernels.push_back(kernel);
	}
	return { (inputDimensions.width - kernelSize + 2 * padding) / stride + 1, (inputDimensions.height - kernelSize + 2 * padding) / stride + 1, kernelSize };
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
//	outputDimensions = { out[0].cols(), out[0].rows(), out.size() };
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

dimensionData ActivationLayer::connectLayerVirtual()
{
	return inputDimensions;
}

PoolingLayer::PoolingLayer(size_t size, size_t stride) : size(size), stride(stride)
{
}

std::vector<Matrix<>> PoolingLayer::calculate(std::vector<Matrix<>>& inputs)
{
	temp = inputs;
	std::vector<Matrix<>> out;
	for (Matrix<> & m : inputs)
		out.push_back(m.maxPool(size, stride));
//	outputDimensions = { out[0].cols(), out[0].rows(), out.size() };
	return out;
}

/**
* The gradients are passed only to the elements that were maximums and therefore only to the elements that had an impact on the final result
* All others have a gradient of 0
*/
std::vector<Matrix<>> PoolingLayer::backprop(std::vector<Matrix<>>& costs)
{
	if(temp.size() != costs.size()) throw UndefinedException(ERR_STR("Input and output dimensions don't match"));
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < temp.size(); ++i) {
		Matrix<> buf(temp[i].rows(), temp[i].cols());
		size_t k = 0;
		for (size_t x = 0; x < temp[i].cols(); x += stride) {
			for (size_t y = 0; y < temp[i].rows(); y += stride) {
				std::pair<size_t, size_t> maxCoords;
				double max = FLT_MIN;
				for (size_t x1 = 0; x1 < size; ++x1) {
					for (size_t y1 = 0; y1 < size; ++y1) {
						if (temp[i](y + y1, x + x1) > max) {
							max = temp[i](y + y1, x + x1);
							maxCoords = std::make_pair(y + y1, x + x1);
						}
					}
				}
				buf(maxCoords.first, maxCoords.second) = costs[i][k++];
			}
		}
		out.push_back(buf);
	}
	return out;
}

dimensionData PoolingLayer::connectLayerVirtual()
{
	return {(inputDimensions.width - size) / stride + 1, (inputDimensions.height - size) / stride + 1, inputDimensions.depth};
}

FCLayer::FCLayer(size_t outputSize) : outputSize(outputSize)
{
	biases.resize(outputSize);
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
//	outputDimensions = { out[0].cols(), out[0].rows(), out.size() };
	return out;
}

std::vector<Matrix<>> FCLayer::backprop(std::vector<Matrix<>>& costs)
{
	if(costs.size() != 1) throw UndefinedException(ERR_STR("Backprop on fully connected layer is expected to be a vector"));
	//dCdB is costs[0]
	Matrix<> dCdW = costs[0] * transpose(temp[0]);
	biases -= learningRate * static_cast<Vector<>>(costs[0]);
	weights -= learningRate * dCdW;

	std::vector<Matrix<>> out;
	out.push_back(transpose(weights) * costs[0]);
	if (previous && previous->getOutputDimensions().width != 1)
	{
		return unflatten(out[0], previous->getOutputDimensions());
	}
	return out;
}

dimensionData FCLayer::connectLayerVirtual()
{
	weights.resize(outputSize, inputDimensions.depth * inputDimensions.width * inputDimensions.height);
	randomize(weights);
	return {1, outputSize, 1};
}

