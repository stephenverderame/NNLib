#pragma once
#include "Matrix.h"
#include <vector>
#include <functional>
using Activation = std::function<Matrix<>(const Matrix<> &)>;
class Layer
{
protected:
	std::vector<Matrix<>> temp;
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
	std::vector<Matrix<>> kernels;
	std::vector<double> biases;
	size_t stride, padding;
public:
	ConvLayer(size_t size, size_t depth, size_t stride, size_t padding);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
class ActivationLayer : public Layer {
private:
	Activation function, functionPrime;
public:
	ActivationLayer(Activation f, Activation fP);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
class PoolingLayer : public Layer {
private:
	size_t size, stride;
public:
	PoolingLayer(size_t size, size_t stride);
	std::vector<Matrix<>> calculate(std::vector<Matrix<>> & inputs) override;
	std::vector<Matrix<>> backprop(std::vector<Matrix<>> & costs) override;
	std::vector<Matrix<>> getWeights() override;
};
class FCLayer : public Layer {
private:
	size_t inputSize, outputSize;
	Matrix<> weights;
	Vector<> biases;
	double learningRate;
public:
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


#define New(T, ...) std::make_unique<T>(__VA_ARGS__)
