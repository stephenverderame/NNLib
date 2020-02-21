#pragma once
#include "NeuralNetwork.h"
#include "Matrix.h"
;
class FeedForward : public NeuralNetwork
{
	friend NetworkSerializer;
private:
	std::vector<size_t> layers;
	std::vector<Matrix<>> weights;
	std::vector<Vector<>> biases;
	mutable std::vector<Vector<>> activations;
	mutable std::vector<Vector<>> z;
	std::vector<double> learningRates;
private:
	void initMatrices();
	void randomize(Matrix<> & mat);
	void randomize(Vector<> & vec);
public:
	FeedForward(std::initializer_list<size_t> layers);
	FeedForward(size_t numLayers, size_t * layers);
	FeedForward(size_t numLayers, ...);
	void resize(std::vector<size_t> & layers);
	Vector<> calculate(const Vector<> & input) const override;
	void backprop(const Vector<> & out, const Vector<> & real) override;
	void setLearningRates(double r0, ...);
	void trainMultithreaded(const std::vector<Vector<>> & inputs, std::function<Vector<>(const Vector<> & x, size_t id)> calcReal, uint8_t threads, size_t batchSize, size_t totalIterations);
#ifdef _GPGPU
	void trainGPU(const std::vector<Vector<>> & inputs, const std::vector<Vector<>> & outputs, size_t batchSize, size_t totalIterations);
#endif
public:
	static void threadHelper(FeedForward & nn, const std::vector<Vector<>> & inputs, std::function<Vector<>(const Vector<> & x, size_t id)> & calcReal, size_t iterations);
};
double vectorSum(Vector<> & v);
double vectorSum(Vector<> && v);

