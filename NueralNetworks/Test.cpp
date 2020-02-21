#include "Matrix.h"
#include <iostream>
#include "FeedForward.h"
#include <time.h>
#include "MNIST.h"
#include "AdvancedFeedForward.h"
#define relu(x) ((x) >= 0 ? (x) : ((x) / 20.0))
#define reluP(x) ((x) >= 0 ? 1 : (1.0 / 20.0))
#define sig(x) (1.0 / (1 + exp(-(x))))
#define sigP(x) (sig(x) * (1 - sig(x)))
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
	convData c(5, 2, 1, 10);
	activationData a(reLu, d_reLu);
	poolData p(2, 2);
	fcData fc(/*6760 26 x 26 x 10*/ 1690, 10);
//	AdvancedFeedForward net(28, 1, 10, {&c, &a, &c, &a, &p, &c, &a, &c, &a, &fc});
	AdvancedFeedForward net(28, 1, 10, { &c, &a, &p, &fc });
	Matrix<> test(28, 28);
	randomize(test);
	Vector<> out = net.calculate((Vector<>)test);
	std::cout << out << std::endl;
	std::cout << "Size: " << out.size() << std::endl;
	getchar();
	return 0;
}