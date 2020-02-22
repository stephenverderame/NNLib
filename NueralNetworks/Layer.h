#pragma once
#include "Matrix.h"
#include <vector>
#include <functional>
#include "NeuralNetwork.h"
using Activation = std::function<Matrix<>(const Matrix<> &)>;
class Layer
{
protected:
	std::vector<Matrix<>> temp;
	//Pointer to previous layer. Assumed to have an equal lifetime as this layer. Not a resource of Layer class
	Layer * previous;
public:
	Layer();
	~Layer();
	void setPreviousLayer(Layer * p);
public:
	virtual std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) = 0;
	virtual std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) = 0;
	virtual std::vector<Matrix<>> getWeights() = 0;
};
class ConvLayer : public Layer {
private:
	std::vector<std::vector<Matrix<>>> kernels;
	//each kernel has a bias that is added to the result of every dot product
	std::vector<double> biases;
	size_t stride, padding;
	bool parameterSharing;
public:
	/**
	* @param size, the size of the convolution kernels
	* @param kernels, the amount of convolution filters/kernels. Each kernel has a depth of the input
	* @param stride, the stride of each kernel
	* @param padding, the zero padding applied to the image. Don't set to automatically use a zero padding that prevents downsizing
	* The convolution layer has k kernels, for each element in the input matrix, the dot product is computed and becomes an element in the output matrix
	* The depth of the output matrix is equal to the amount of kernels
	* @param paramSharing, kernels share weights for every depth. In practice this reduces the amount of kernels bc each kernel depth does not have its own unique set of weights
	*  dot product at each depth and sum the result for a final output value
	*/
	ConvLayer(size_t size, size_t kernelNum, size_t depth, size_t stride = 1, size_t padding = -1, bool paramSharing = true);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
class ActivationLayer : public Layer {
private:
	Activation function, functionPrime;
public:
	//Activation function and the derivative of the activation
	ActivationLayer(Activation f, Activation fP);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
class PoolingLayer : public Layer {
private:
	size_t size, stride;
public:
	/**
	* @param size, square width of pooling filter.
	* @param stride, how far to move the pooling filter after each pass
	* The pooling layer creates a filter of size x size and lines it up with a size x size submatrix in the matrix
	* performs an operation and returns a single value from the size x size submatrix as a result to go into the resulting matrix
	* this downsizes the input matrix. A size of 2 and stride of 2 downsizes the matrix by half
	*/
	PoolingLayer(size_t size = 2, size_t stride = 2);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
//Fully Connected Neural Network Layer
//Linearizes the matrix input to a vector
class FCLayer : public Layer {
private:
	size_t inputSize, outputSize;
	Matrix<> weights;
	Vector<> biases;
	double learningRate;
public:
	/**
	* @param inputSize, total size of input matrix
	* @param outputSize, totalSize of output vector
	*/
	FCLayer(size_t inputSize, size_t outputSize);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};

//template<typename T> 
//const auto& New = std::make_unique<T>;

static const Activation reLu = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = m.get(i) >= 0 ? m.get(i) : 0;
	return o;
};
static const Activation d_reLu = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = m.get(i) >= 0 ? 1 : 0;
	return o;
};
static const Activation sig_m = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = sig(m.get(i));
	return o;
};
static const Activation d_sig_m = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = sigP(m.get(i));
	return o;
};
static const Activation fsig_m = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = f_sig(m.get(i));
	return o;
};
static const Activation d_fsig_m = [](const Matrix<> & m) -> Matrix<> {
	Matrix<> o(m.rows(), m.cols());
	for (size_t i = 0; i < m.size(); ++i)
		o[i] = f_sigP(m.get(i));
	return o;
};


#define New(T, ...) std::make_unique<T>(__VA_ARGS__)
