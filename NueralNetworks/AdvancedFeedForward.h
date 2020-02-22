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
struct convData : public layerData {
	size_t filterSize, padding, stride, filters, depth;
	bool sharing;
	//if stride is 1, padding can be set to -1 to autocalculate valid padding
	convData(size_t fSize, size_t pad, size_t stride, size_t filters, size_t inputDepth, bool sharing) : layerData(layer_t::conv), filterSize(fSize), padding(pad), stride(stride), 
		filters(filters), depth(inputDepth), sharing(sharing) {};
};
struct poolData : public layerData {
	size_t poolSize, stride;
	poolData(size_t fSize, size_t stride) : layerData(layer_t::pooling), poolSize(fSize), stride(stride) {};
};
struct activationData : public layerData {
	Activation function, derivative;
	activationData(const Activation & f, const Activation & d_f) : layerData(layer_t::activation), function(f), derivative(d_f) {};
};
struct fcData : public layerData {
	size_t isize, osize;
	fcData() : layerData(layer_t::fullyConnected), isize(0), osize(0) {};
	fcData(size_t inputSize, size_t outputSize) : layerData(layer_t::fullyConnected), isize(inputSize), osize(outputSize) {};
};
//TMP helpers for layer_cast(). 
template<typename T, typename = void>
struct is_conv : std::false_type {};
template<typename T>
struct is_conv<T, std::void_t<decltype(std::declval<T&>().filters)>> : std::true_type {};
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
	size_t inSize, inDepth, outSize;
public:
	AdvancedFeedForward();
	AdvancedFeedForward(size_t inputSize, size_t inputDepth, size_t outputSize, std::initializer_list<layerData *> layers);
	~AdvancedFeedForward();
	Vector<> calculate(const Vector<> & input) const override;
	void backprop(const Vector<> & out, const Vector<> & real) override;
public:
	static Vector<> m2v(std::vector<Matrix<>> mat3d);
	static std::vector<Matrix<>> v2m(const Vector<> & v, size_t width, size_t height, size_t depth);
};

