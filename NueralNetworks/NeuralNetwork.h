#pragma once
#include "Vector.h"
#include <functional>
#include <memory>
using ActivationFunction = std::function < Vector<>(const Vector<> &)>;
enum NetworkTypes : uint16_t {
	nt_null = 0,
	nt_feedForward = 'FF'
};
class NeuralNetwork
{
	friend class NetworkSerializer;
protected:
	double learningRate = 0.7;
	ActivationFunction f, fPrime;
public:
	void setLearningRate(double r) { learningRate = r; }
	virtual void setActivation(const ActivationFunction & f, const ActivationFunction & fPrime) {
		this->f = f; this->fPrime = fPrime;
	}
	virtual Vector<> calculate(const Vector<> & input) const = 0;
	virtual void backprop(const Vector<> & out, const Vector<> & real) = 0;
};
class NetworkSerializer {
private:
	static std::unique_ptr<NeuralNetwork> loadFF(FILE * f);
	static void saveFF(FILE * f, NeuralNetwork * nn);
public:
	static std::unique_ptr<NeuralNetwork> loadNet(const char * filename);
	static void saveNet(NeuralNetwork * nn, NetworkTypes type, const char * filename);
};
const static ActivationFunction leakyReLu = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = v.get(index) >= 0 ? v.get(index) : v.get(index) / 20.0;
		++index;
	case 3:	out[index] = v.get(index) >= 0 ? v.get(index) : v.get(index) / 20.0;
		++index;
	case 2:	out[index] = v.get(index) >= 0 ? v.get(index) : v.get(index) / 20.0;
		++index;
	case 1:	out[index] = v.get(index) >= 0 ? v.get(index) : v.get(index) / 20.0;
		++index;
		} while (index < out.size());
	}
	return out;
};
const static ActivationFunction leakyReLuP = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = v.get(index) >= 0 ? 1 : 1 / 20.0;
		++index;
	case 3:	out[index] = v.get(index) >= 0 ? 1 : 1 / 20.0;
		++index;
	case 2: out[index] = v.get(index) >= 0 ? 1 : 1 / 20.0;
		++index;
	case 1:	out[index] = v.get(index) >= 0 ? 1 : 1 / 20.0;
		++index;
		} while (index < out.size());
	}
	return out;
};
const static auto sig = [](double x) -> double {
	return 1.0 / (1 + exp(-x));
};
const static ActivationFunction sigmoid = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = sig(v.get(index));
		++index;
	case 3:	out[index] = sig(v.get(index));
		++index;
	case 2: out[index] = sig(v.get(index));
		++index;
	case 1: out[index] = sig(v.get(index));
		++index;
		} while (index < out.size());
	}
	return out;
};
const static ActivationFunction sigmoidP = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = sig(v.get(index)) * (1 - sig(v.get(index)));
		++index;
	case 3:	out[index] = sig(v.get(index)) * (1 - sig(v.get(index)));
		++index;
	case 2: out[index] = sig(v.get(index)) * (1 - sig(v.get(index)));
		++index;
	case 1:	out[index] = sig(v.get(index)) * (1 - sig(v.get(index)));
		++index;
		} while (index < out.size());
	}
	return out;
};

const static ActivationFunction ReLu = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = v.get(index) >= 0 ? v.get(index) : 0;
		++index;
	case 3:	out[index] = v.get(index) >= 0 ? v.get(index) : 0;
		++index;
	case 2:	out[index] = v.get(index) >= 0 ? v.get(index) : 0;
		++index;
	case 1:	out[index] = v.get(index) >= 0 ? v.get(index) : 0;
		++index;
	} while (index < out.size());
	}
	return out;
};
const static ActivationFunction ReLuP = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
			out[index] = v.get(index) >= 0 ? 1 : 0;
		++index;
	case 3:	out[index] = v.get(index) >= 0 ? 1 : 0;
		++index;
	case 2: out[index] = v.get(index) >= 0 ? 1 : 0;
		++index;
	case 1:	out[index] = v.get(index) >= 0 ? 1 : 0;
		++index;
	} while (index < out.size());
	}
	return out;
};
#define f_sig(x) ((x) / (1.0 + abs(x)))
#define f_sigP(x) (1.0 / ((abs(x) + 1) * (abs(x) + 1)))
const static ActivationFunction f_sigmoid = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
		out[index] = f_sig(v.get(index));
		++index;
	case 3:	out[index] = f_sig(v.get(index));
		++index;
	case 2: out[index] = f_sig(v.get(index));
		++index;
	case 1: out[index] = f_sig(v.get(index));
		++index;
	} while (index < out.size());
	}
	return out;
};
const static ActivationFunction f_sigmoidP = [](const Vector<> & v) -> Vector<> {
	Vector<> out(v.size());
	size_t index = 0;
	switch (out.size() % 4) {
	case 0: do {
		out[index] = f_sigP(v.get(index));
		++index;
	case 3:	out[index] = f_sigP(v.get(index));
		++index;
	case 2: out[index] = f_sigP(v.get(index));
		++index;
	case 1: out[index] = f_sigP(v.get(index));
		++index;
	} while (index < out.size());
	}
	return out;
};

