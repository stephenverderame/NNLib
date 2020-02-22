#include "AdvancedFeedForward.h"

template<typename T>
const std::function<T*(layerData*)> castHelper<T>::retTable[] = {
		[](layerData* ptr) -> T* {return is_conv<T>::value ? static_cast<T*>(ptr) : nullptr; },
		[](layerData* ptr) -> T* {return is_pool<T>::value ? static_cast<T*>(ptr) : nullptr; },
		[](layerData* ptr) -> T* {return is_act<T>::value ? static_cast<T*>(ptr) : nullptr; },
		[](layerData* ptr) -> T* {return is_fc<T>::value ? static_cast<T*>(ptr) : nullptr; }
};

AdvancedFeedForward::AdvancedFeedForward()
{
}

AdvancedFeedForward::AdvancedFeedForward(size_t inS, size_t inD, size_t outS, std::initializer_list<layerData *> layers) : inSize(inS), inDepth(inD), outSize(outS)
{
	for (layerData * l : layers) {
		if (convData * c = layer_cast<convData>(l))
			this->layers.push_back(New(ConvLayer, c->filterSize, c->filters, c->depth, c->stride, c->padding, c->sharing));
		else if (poolData * p = layer_cast<poolData>(l))
			this->layers.push_back(New(PoolingLayer, p->poolSize, p->stride));
		else if (activationData * a = layer_cast<activationData>(l))
			this->layers.push_back(New(ActivationLayer, a->function, a->derivative));
		else if (fcData * f = layer_cast<fcData>(l))
			this->layers.push_back(New(FCLayer, f->isize, f->osize));
	}
	for (size_t i = this->layers.size() - 1; i > 0; --i)
		this->layers[i]->setPreviousLayer(this->layers[i - 1].get());
}


AdvancedFeedForward::~AdvancedFeedForward()
{
}

Vector<> AdvancedFeedForward::calculate(const Vector<>& input) const
{
	std::vector<Matrix<>> temp = v2m(input, inSize, inSize, inDepth);
	for (size_t i = 0; i < layers.size(); ++i)
		temp = layers[i]->calculate(temp);
	return m2v(temp);
}

void AdvancedFeedForward::backprop(const Vector<>& out, const Vector<>& real)
{
}

Vector<> AdvancedFeedForward::m2v(std::vector<Matrix<>> mat3d)
{
	Vector<> out(mat3d.size() * mat3d[0].size());
	for (size_t i = 0; i < mat3d.size(); ++i) {
		for (size_t j = 0; j < mat3d[i].size(); ++j)
			out[i * mat3d[0].size() + j] = mat3d[i][j];
	}
	return out;
}

std::vector<Matrix<>> AdvancedFeedForward::v2m(const Vector<>& v, size_t width, size_t height, size_t depth)
{
	std::vector<Matrix<>> out;
	for (size_t i = 0; i < depth; ++i) {
		Matrix<> m(height, width);
		for (size_t j = 0; j < width * height; ++j)
			m[j] = v.get(i * width * height + j);
		out.push_back(m);
	}
	return out;
}
