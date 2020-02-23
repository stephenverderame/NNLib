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

AdvancedFeedForward::AdvancedFeedForward(dimensionData inS, dimensionData outS, std::initializer_list<layerData *> layers) : inSize(inS), outSize(outS)
{
	for (layerData * l : layers) {
		switch (l->type) {
		case layer_t::conv:
		{
			auto c = static_cast<convData*>(l);
			this->layers.push_back(New(ConvLayer, c->kernelSize, c->kernels, c->stride, c->padding, c->sharing));
			break;
		}
		case layer_t::pooling:
		{
			auto p = static_cast<poolData*>(l);
			this->layers.push_back(New(PoolingLayer, p->poolSize, p->stride));
			break;
		}
		case layer_t::activation:
		{
			auto a = static_cast<activationData*>(l);
			this->layers.push_back(New(ActivationLayer, a->function, a->derivative));
			break;
		}
		case layer_t::fullyConnected:
		{
			auto f = static_cast<fcData*>(l);
			this->layers.push_back(New(FCLayer, f->osize));
			break;
		}
		}
	}
	connectLayers();
}


AdvancedFeedForward::~AdvancedFeedForward()
{
}

Vector<> AdvancedFeedForward::calculate(const Vector<>& input) const
{
	std::vector<Matrix<>> temp = v2m(input, inSize.width, inSize.height, inSize.depth);
	for (size_t i = 0; i < layers.size(); ++i)
		temp = layers[i]->calculate(temp);
	return m2v(temp);
}

void AdvancedFeedForward::backprop(const Vector<>& out, const Vector<>& real)
{
	Matrix<> gradient = static_cast<Matrix<>>(2.0 * (real - out));
	std::vector<Mat> grads;
	grads.push_back(gradient);
	for (size_t i = layers.size() - 1; i >= 0; --i) {
		layers[i]->backprop(grads);
	}
}

void AdvancedFeedForward::setLearningRates(std::initializer_list<double> rates)
{
	assert((rates.size() == layers.size() && rates.size() > 1) && "Each layer must have a learning rate!");
	if (rates.size() == 1) {
		for (auto& l : layers) {
			l->setLearningRate(*rates.begin());
		}
	}
	else {
		size_t i = 0;
		for (auto r : rates) {
			layers[i++]->setLearningRate(r);
		}
	}

}

void AdvancedFeedForward::connectLayers()
{
	dimensionData in = inSize;
	Layer * prev = nullptr;
	for (size_t i = 0; i < layers.size(); ++i) {
		layers[i]->setPreviousLayer(prev);
		prev = layers[i].get();
		in = layers[i]->connectLayer(in);
	}
	assert(in.depth == outSize.depth && in.height == outSize.height && in.width == outSize.width && "Output size mismatch");
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
