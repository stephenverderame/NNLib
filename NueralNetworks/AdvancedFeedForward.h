#pragma once
#include "NeuralNetwork.h"
#include "Layer.h"
#include <memory>
enum class layer_t {
	conv, pooling, activation, fullyConnected, 
	count //value representing the amount of types
};
struct layerData {
	layer_t type;
	layerData(layer_t t) : type(t) {};
};
//Hyperparameters for a convolutional layer
struct convData : public layerData {
	size_t kernelSize, padding, stride, kernels;
	bool sharing;
	//if stride is 1, padding can be set to -1 to autocalculate valid padding
	convData(size_t kSize, size_t kernels, bool sharing, size_t stride = 1, size_t pad = defaultVal) : layerData(layer_t::conv), kernelSize(kSize), padding(pad), stride(stride),
		kernels(kernels), sharing(sharing) {};
};
//Hyperparameters for a pooling layer
struct poolData : public layerData {
	size_t poolSize, stride;
	poolData(size_t fSize = 2, size_t stride = 2) : layerData(layer_t::pooling), poolSize(fSize), stride(stride) {};
};
//Hyperparameters for an activation layer
struct activationData : public layerData {
	Activation function, derivative;
	activationData(const Activation & f, const Activation & d_f) : layerData(layer_t::activation), function(f), derivative(d_f) {};
};
//Hyperparameters for a fully-connected layer
struct fcData : public layerData {
	size_t osize;
	fcData() : layerData(layer_t::fullyConnected), osize(0) {};
	fcData(size_t outputSize) : layerData(layer_t::fullyConnected), osize(outputSize) {};
};
//TMP helpers for layer_cast(). 
template<typename T, typename = void>
struct is_conv : std::false_type {};
template<typename T>
struct is_conv<T, std::void_t<decltype(std::declval<T&>().kernels)>> : std::true_type {};
template<typename T, typename = void>
struct is_pool : std::false_type {};
template<typename T>
struct is_pool<T, std::void_t<decltype(std::declval<T&>().poolSize)>> : std::true_type {};
template<typename T, typename = void>
struct is_act : std::false_type {};
template<typename T>
struct is_act<T, std::void_t<decltype(std::declval<T&>().derivative)>> : std::true_type {};
template<typename T, typename = void>
struct is_fc : std::false_type {};
template<typename T>
struct is_fc<T, std::void_t<decltype(std::declval<T&>().osize)>> : std::true_type {};


template<typename T>
class castHelper {
//	friend std::unique_ptr<T> layer_cast(std::unique_ptr<layerData>);

	/**
	* This is essentially a jump table to optimize the layer cast. Uses the layer_t type in all layerData objects to index and call the correct TMP type check
	*/
	const static std::function<T*(layerData*)> retTable[static_cast<size_t>(layer_t::count)];
public:
	static T * layerCast(layerData * ptr) {
		if (static_cast<size_t>(ptr->type) >= 0 && static_cast<size_t>(ptr->type) < static_cast<size_t>(layer_t::count))
			return retTable[static_cast<size_t>(ptr->type)](ptr);
		return nullptr;
	}
};

/**
* Casts the passed layer data pointer to a layerData pointer of type T
* If the passed pointer is not of type T, returns nullptr
*/
template<typename T>
const auto layer_cast = castHelper<T>::layerCast;


class AdvancedFeedForward : public NeuralNetwork
{
	std::vector<std::unique_ptr<Layer>> layers;
	dimensionData inSize, outSize;
public:
	AdvancedFeedForward();
	/**
	* @param layers, the addresses or managed pointers of layerData structs in the order they should be
	*   each struct contains the hyperparameters for each layer
	*/
	AdvancedFeedForward(dimensionData inputDims, dimensionData outputSize, std::initializer_list<layerData *> layers);
	~AdvancedFeedForward();
	Vector<> calculate(const Vector<> & input) const override;
	void backprop(const Vector<> & out, const Vector<> & real) override;

	/**Sets the learning rates for all of the networks layers
	* @precondition, rates.size() = numLayers or rates.size() = 1
	* If only 1 rate is passed, this is the learning rate for the entire network
	*/
	void setLearningRates(std::initializer_list<double> rates);

	/**
	* Sets the previous layer for each network layer, finishes layer setup. Called in constructor
	*/
	void connectLayers();
public:
	//Matrix to vector
	static Vector<> m2v(std::vector<Matrix<>> mat3d);
	//Vector to matrix
	static std::vector<Matrix<>> v2m(const Vector<> & v, size_t width, size_t height, size_t depth);
};

