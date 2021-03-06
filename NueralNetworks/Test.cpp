#include "Matrix.h"
#include <iostream>
#include "FeedForward.h"
#include <time.h>
#include "MNIST.h"
#include "AdvancedFeedForward.h"
#include "Layer.h"
/*
#define relu(x) ((x) >= 0 ? (x) : ((x) / 20.0))
#define reluP(x) ((x) >= 0 ? 1 : (1.0 / 20.0))
#define sig(x) (1.0 / (1 + exp(-(x))))
#define sigP(x) (sig(x) * (1 - sig(x)))
*/
int main() {
/*	auto ff = NetworkSerializer::loadNet("xorModel.net");
	ff->setActivation(sigmoid, sigmoidP); 
	iterations = 100;
	for (int i = 0; i < iterations; ++i) {
		Vector<> input = { round((double)rand() / RAND_MAX), round((double)rand() / RAND_MAX) };
		Vector<> out = ff->calculate(input);
		int real = ((int)input[0] ^ (int)input[1]);
		if ((int)round(out[0]) == real) ++correct;
	}
	printf("Correct: %d / %d \n", correct, iterations);
//	NetworkSerializer::saveNet(ff.get(), nt_feedForward, "xorModel.net");*/

#ifdef _training
	auto images = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\train-images.idx3-ubyte");
	auto labels = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\train-labels.idx1-ubyte");
	auto testImages = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-images.idx3-ubyte");
	auto testLabels = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-labels.idx1-ubyte");
	FeedForward nn(4, 28 * 28, 16, 16, 10);
	nn.setActivation(sigmoid, sigmoidP);
	nn.setLearningRate(0.1);
/*	for (int i = 0; i < iterations; ++i) {
		int n = i % images.size();
		Vector<> input = images[n];
		assert(input.size() == 28 * 28);
		Vector<> output = nn.calculate(input);

		Vector<> real(10);
		real.zero();
		real[labels[n]] = 1.0;
		nn.backprop(output, real);
		if(i % 100) printf("Training: %.2f%%\r", (double)i / iterations * 100.0);
	}
	nn.trainMultithreaded(images, [&labels](const Vector<> & x, size_t id) -> Vector<> {
		Vector<> r(10);
		r.zero();
		r[labels[id]] = 1.0;
		return r;
	}, 6, 10000, images.size() * 1000);
	printf("\n");
	NetworkSerializer::saveNet(&nn, nt_feedForward, "mnistModel2.net");
	assert(images.size() == labels.size() && images.size() > 0);
	correct = 0;
	iterations = 1000;
	for (int i = 0; i < iterations; ++i) {
		Vector<> x = (Vector<>)images[i];
		Vector<> y = nn.calculate(x);
		double max = DBL_MIN;
		int id = -1;
		for (int i = 0; i < y.size(); ++i) {
			if (y[i] > max) {
				max = y[i];
				id = i;
			}
		}
		if (id == labels[i]) ++correct;
		printf("Testing: %.2f%%\r", (double)i / iterations * 100.0);
	}
	printf("Testing: 100.00%%\n");
	printf("Correct: %d / %d\n", correct, iterations);
	*/
	size_t epochs = 1000;
	for (size_t i = 0; i < epochs; ++i) {
		nn.trainMultithreaded(images, [&labels](const Vector<> & x, size_t id) -> Vector<> {
			Vector<> r(10);
			r.zero();
			r[labels[id]] = 1.0;
			return r;
		}, 6, 10000, images.size());
		int correct = 0;
		int testSize = testImages.size();
		for (int j = 0; j < testSize; ++j) {
			Vector<> x = (Vector<>)testImages[j];
			Vector<> y = nn.calculate(x);
			double max = DBL_MIN;
			int id = -1;
			for (int k = 0; k < y.size(); ++k) {
				if (y[k] > max) {
					max = y[k];
					id = k;
				}
			}
			if (id == testLabels[i]) ++correct;
		}
		float r = (float)correct / testSize;
		printf("Epoch %d: Correct: %d / %d (%.2f%%)\n", i, correct, testSize, r * 100.f);
		if(r >= 0.8) NetworkSerializer::saveNet(&nn, nt_feedForward, "mnistModel_g.net");
	}
	NetworkSerializer::saveNet(&nn, nt_feedForward, "mnistModel.net");
#endif
/*	auto net = NetworkSerializer::loadNet("mnistModel_g.net");
	net->setActivation(sigmoid, sigmoidP);
	auto testImages = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-images.idx3-ubyte");
	auto testLabels = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-labels.idx1-ubyte");
	int correct = 0, iterations = 10000;
	for (int i = 0; i < iterations; ++i) {
		int id = i;
		Vector<> x = testImages[id];
		Vector<> y = net->calculate(x);
		int actual = testLabels[id];
		int calc = -1;
		double mx = DBL_MIN;
		for (int j = 0; j < y.size(); ++j) {
			if (y[j] > mx) {
				mx = y[j];
				calc = j;
			}
		}
		if (actual == calc) ++correct;
	}
	printf("%d / %d correct! (%.2f%%)\n", correct, iterations, (float)correct / iterations * 100.f);
	/////////////////////////////////////////////////
	auto images = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\train-images.idx3-ubyte");
	auto labels = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\train-labels.idx1-ubyte");
	auto testImages = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-images.idx3-ubyte");
	auto testLabels = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-labels.idx1-ubyte");
	FeedForward ff({ 28 * 28, 2500, 2000, 1500, 1000, 500, 10 });
	ff.setActivation(ReLu, ReLuP);
	ff.setLearningRate(0.001);
	ff.trainMultithreaded(images, [&](const Vector<> & x, size_t id) -> Vector<> {
		Vector<> r(10);
		r.zero();
		r[labels[id]] = 1.0;
		return r;
	}, 6, 10000, 5 * images.size());
	int correct = 0, iterations = 10000;
	for (int i = 0; i < iterations; ++i) {
		int id = i % testImages.size();
		Vector<> x = testImages[id];
		Vector<> y = ff.calculate(x);
		int actual = testLabels[id];
		int calc = -1;
		double mx = DBL_MIN;
		for (int j = 0; j < y.size(); ++j) {
			if (y[j] > mx) {
				mx = y[j];
				calc = j;
			}
		}
		if (actual == calc) ++correct;
	}
	printf("%d / %d correct! (%.2f%%)\n", correct, iterations, (float)correct / iterations * 100.f);
	NetworkSerializer::saveNet(&ff, nt_feedForward, "bigMnistModel.net");
	*/
	convData c(5, 10, true);
//	convData c2(5, 2, 1, 10, 10, true);
	activationData a(reLu, d_reLu);
	poolData p;
	fcData fc(10);
	activationData fa(fsig_m, d_fsig_m);
	AdvancedFeedForward net({ 28, 28, 1 }, {1, 10, 1}, { &c, &fc, &fa});
/*	Matrix<> test(28, 28);
	randomize(test);
//	Vector<> out = net.calculate((Vector<>)test);
	std::cout << out << std::endl;
	std::cout << "Size: " << out.size() << std::endl;
	Vector<> testReal(10);
	testReal[3] = 1.0;
//	net.backprop(out, testReal);
	printf("Done!\n");
	printf("Read\n");*/
	auto img = mnist::readImages("C:\\Users\\stephen\\Downloads\\MNIST\\t10k-images.idx3-ubyte");
	auto lbl = mnist::readLabels("C:\\Users\\stephen\\Downloads\\MNIST\\train-labels.idx1-ubyte");
	Vector<> real2(10);
	Vector<> test2;
	net.setLearningRates({ 0.7 });
	size_t correct = 0;
	for (int i = 0; i < 1000; ++i) {
		test2 = img[i % img.size()];
		real2.zero();
		real2[lbl[i % lbl.size()]] = 1.0;
		Vector<> out2 = net.calculate(test2);
		printf("Cost: %f\n", dot(out2 - real2, out2 - real2));
		net.backprop(out2, real2);
		if (getMaxIndex(out2) == lbl[i % lbl.size()]) ++correct;
		if (i % 10 == 0) printf("%d / %d Correct!\n", correct, i);
	}
	printf("%d / 1000\n", correct);
	printf("Done\n");
/*	convData tC(3, 2, true);
	fcData tF(1);
	AdvancedFeedForward tNet({ 1, 1, 1 }, { 1, 1, 1 }, {&tC, &tC, &tF, &fa});
	for (int i = 0; i < 200; ++i) {
		Vector<> in = { (double)rand() / RAND_MAX };
		Vector<> r = { 1.0 };
		auto out = tNet.calculate(in);
		printf("Out: %f Cost: %f\n", out[0], hadamard(out - r, out - r)[0]);
		tNet.backprop(out, r);
	}*/
	getchar();
	return 0;
}