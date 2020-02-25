#pragma once
#include "Vector.h"
template<typename T = double>
class Matrix {
private:
	std::vector<T> data;
	size_t _rows, _cols;
public:
	Matrix() : _rows(0), _cols(0) {};
	Matrix(size_t row, size_t col) : _rows(row), _cols(col) {
		data.resize(row * col, 0);
	}
	Matrix(std::initializer_list<std::vector<T>> & list) {
		for (auto v : list) {
			data.insert(data.end(), v.begin(), v.end());
			++_rows;
		}
		_cols = (*list.begin()).size();
	}
	Matrix(const Matrix<T> & other) {
		data = other.data;
		_rows = other._rows;
		_cols = other._cols;
	}
	explicit Matrix(const Vector<T> & v) : _rows(v.size()), _cols(1) {
		data.resize(v.size());
		for (size_t i = 0; i < v.size(); ++i)
			data[i] = v.get(i);
	}
	Matrix<T>& operator=(const Matrix<T> & other) {
		data = other.data;
		_rows = other._rows;
		_cols = other._cols;
		return *this;
	}
	Matrix<T>& operator=(Vector<T> & v) {
		data.clear();
		data.insert(data.end(), v.vbegin(), v.vend());
		_rows = v.size();
		_cols = 1;
		return *this;
	}
	Matrix<T>& operator=(const T & scalar) {
		for (int i = 0; i < data.size(); ++i)
			data[i] = scalar;
	}
	T& operator[](size_t index) throw (UndefinedException) {
		if (index >= data.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return data[index];
	}
	T& operator()(size_t row, size_t col) throw (UndefinedException) {
		if(row * _cols + col >= data.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return data[row * _cols + col];
	}
	void resize(size_t row, size_t col) noexcept {
		data.resize(row * col, 0);
		_rows = row;
		_cols = col;
	}
	Matrix<T>& operator*=(T scalar) {
		size_t index = 0;
		size_t s = data.size();
		switch (s % 4) {
		case 0: do {
				data[index++] *= scalar;
		case 3:	data[index++] *= scalar;
		case 2:	data[index++] *= scalar;
		case 1:	data[index++] *= scalar;
			} while (index < s);
		}
		return *this;
	}
	Matrix<T>& operator+=(T scalar) {
		size_t index = 0;
		size_t s = data.size();
		switch (s % 4) {
		case 0: do {
				data[index++] += scalar;
		case 3:	data[index++] += scalar;
		case 2:	data[index++] += scalar;
		case 1:	data[index++] += scalar;
			} while (index < s);
		}
		return *this;
	}
	//If matrix is not initialized, mat is initialized with 0 to work with [other]
	Matrix<T>& operator-=(const Matrix<T> & other) throw(UndefinedException) {
		if (_rows == 0 && _cols == 0) resize(other.rows(), other.cols());
		else if (_rows != other.rows() && _cols != other.cols()) throw UndefinedException(ERR_STR("Size mismatch"));
		size_t index = 0;
		size_t s = data.size();
		switch (s % 4) {
		case 0: do {
				data[index] -= other.data[index];
			++index;
		case 3:	data[index] -= other.data[index];
			++index;
		case 2:	data[index] -= other.data[index];
			++index;
		case 1:	data[index] -= other.data[index];
			++index;
			} while (index < s);
		}
		return *this;
	}
	//If matrix is not initialized, mat is initialized with 0 to fit with [other]
	Matrix<T>& operator+=(const Matrix<T> & other) throw(UndefinedException) {
		if (_rows == 0 && _cols == 0) resize(other.rows(), other.cols());
		else if (_rows != other.rows() && _cols != other.cols()) throw UndefinedException(ERR_STR("Size mismatch"));
		size_t index = 0;
		size_t s = data.size();
		switch (s % 4) {
		case 0: do {
				data[index] += other.data[index];
			++index;
		case 3:	data[index] += other.data[index];
			++index;
		case 2:	data[index] += other.data[index];
			++index;
		case 1:	data[index] += other.data[index];
			++index;
			} while (index < s);
		}
		return *this;
	}
	size_t rows() const {
		return _rows;
	}
	size_t cols() const {
		return _cols;
	}
	size_t size() const {
		return data.size();
	}
	/**@return the column vector of column index*/
	Vector<T> getVector(size_t index) const throw(UndefinedException) {
		if(index >= _cols) throw UndefinedException(ERR_STR("Index out of bounds"));
		Vector<T> v(data.size());
		size_t i = 0;
		switch (v.size() % 4) {
		case 0: do {
				v[i] = data[i * _cols + index];
			++i;
		case 3:	v[i] = data[i * _cols + index];
			++i;
		case 2:	v[i] = data[i * _cols + index];
			++i;
		case 1:	v[i] = data[i * _cols + index];
			++i;
			} while (i < v.size());
		}
		return v; 
	}
	T get(size_t row, size_t col) const throw(UndefinedException) {
		if(row * _cols + col >= data.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return data[row * _cols + col];
	}
	std::vector<T> getRow(size_t index) const {
		std::vector<T> vec(_cols);
		vec.insert(vec.end(), data.begin() + index * _cols, data.begin() + (index + 1) * _cols);
		return vec;
	}
	T get(size_t index) const throw(UndefinedException) {
		if(index >= data.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return data[index];
	}
	Matrix<T> transpose() {
		Matrix<T> out(_cols, _rows);
		for (size_t i = 0; i < out.rows(); ++i) {
			for (size_t j = 0; j < out.cols(); ++j) {
				out(i, j) = data[j * _cols + i];
			}
		}
		return out;
	}
	explicit operator Vector<T>() const noexcept {
		Vector<T> out(size());
		for (size_t i = 0; i < out.size(); ++i)
			out[i] = get(i);
		return out;
	}
	void loadFromRawData(std::vector<T> & data) {
		assert(this->data.size() == data.size() && "Size mismatch");
		this->data = data;
	}
	std::vector<T> getData() noexcept {
		return data;
	}
	/**
	* Applies a kernel convolution onto the matrix [in] with a stride [stride]
	* @precondition: kernel is odd sized square matrix and fits into kernel
	* convolves the kernel onto the matrix by lining up the kernel with each kernel sized sub-matrix of the matrix
	* the dot product is computed and the result becomes the element in the output matrix which corresponds to the element that lines up with the center of the kernel
	*/
	Matrix<T> applyAsKernel(const Matrix<T> & in, size_t stride) const {
		//this implementation does not handle padding, input should already be zero padded
		const size_t kernelCols = _cols;
		const size_t kernelRows = _rows;
//		assert(kernelCols == kernelRows && "Kernel must be a square matrix");
//		assert(in.cols() == in.rows() && (in.cols() - kernelCols) % stride == 0 && "Input must be square and kernel must fit");
		if (kernelCols != kernelRows && in.cols() != in.rows() && (in.cols() - kernelCols) % stride != 0) throw UndefinedException(ERR_STR("Undefined kernel convolution"));
		Matrix<T> mat((in.rows() - kernelRows) / stride + 1, (in.cols() - kernelCols) / stride + 1);
		size_t k = 0;
		for (size_t i = 0; i < in.size(); i += stride) {
			int x = i % in.cols();
			int y = i / in.cols();
//			if (x - ((kernelCols - 1) / 2) < 0 || x + (kernelCols - 1) >= in.cols() || y - ((kernelRows - 1) / 2) < 0 || y + ((kernelRows - 1) / 2) >= in.rows()) continue;
			if (x + kernelCols >= in.cols() || y + kernelRows >= in.rows()) continue;
			double sum = 0;
			for (int y1 = 0; y1 < kernelRows; ++y1) {
				for (int x1 = 0; x1 < kernelCols; ++x1) {
					if (x + x1 >= 0 && x + x1 < in.cols() && y + y1 >= 0 && y + y1 < in.rows()) {
						sum += get(y1, x1) * in.get(y + y1, x + x1);
					}
				}
			}
			mat[k++] = sum;
		}
		return mat;
	}
	/**
	* Zero pads the matrix by creating a border of thickness [border] around the matrix
	*/
	Matrix<T> zeroPad(size_t border = 1) const {
		//zero pad by 1 for 3x3 filter, 2 for 5x5 and 3 for 7x7 (assuming a stride of 1)
		Matrix<T> out(_rows + 2 * border, _cols + 2 * border);
		for (size_t x = 0; x < _cols; ++x) {
//			if (i < border || i >= _rows + border) continue;
			for (size_t y = 0; y < _rows; ++y) {
//				if (j < border || j >= _cols + border) continue;
				out(y + border, x + border) = get(y, x);
			}
		}
		return out;
	}
	//*@See PoolingLayer
	Matrix<T> maxPool(size_t kernelSize, size_t stride) const {
		Matrix<T> mat((_rows - kernelSize) / stride + 1, (_cols - kernelSize) / stride + 1);
		size_t k = 0;
		for (int x = 0; x < _cols; x += stride) {
			for (int y = 0; y < _rows; y += stride) {
				T max = std::numeric_limits<T>::min();
				for (int x1 = 0; x1 < kernelSize; ++x1) {
					for (int y1 = 0; y1 < kernelSize; ++y1) {
						if (x + x1 >= 0 && x + x1 < _cols && y + y1 >= 0 && y + y1 < _rows) {
							if (get(y + y1, x + x1) > max)
								max = get(y + y1, x + x1);
						}
					}
				}
				mat[k++] = max;
			}
		}
		return mat;
	}
	/**
	* Rotates the matrix by 180 degrees. This essentially reverses the order of elements
	*/
	Matrix<T> rotate180() const {
		Matrix<T> out(_rows, _cols);
		out.data.insert(out.data.end(), data.rbegin(), data.rend());
		return out;
	}

};
template<typename T>
Matrix<T> operator*(const Matrix<T> & m, const T scalar) {
	Matrix<T> out(m.rows(), m.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m.get(index) * scalar;
		++index;
	case 3:	out[index] = m.get(index) * scalar;
		++index;
	case 2:	out[index] = m.get(index) * scalar;
		++index;
	case 1:	out[index] = m.get(index) * scalar;
		++index;
		} while (index < s);
	}
	return out;
}
template<typename T>
Matrix<T> operator*(const T scalar, const Matrix<T> & m) {
	Matrix<T> out(m.rows(), m.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m.get(index) * scalar;
		++index;
	case 3:	out[index] = m.get(index) * scalar;
		++index;
	case 2: out[index] = m.get(index) * scalar;
		++index;
	case 1:	out[index] = m.get(index) * scalar;
		++index;
		} while (index < s);
	}
	return out;
}
template<typename T>
Matrix<T> operator+(const Matrix<T> & m, const T scalar) {
	Matrix<T> out(m.rows(), m.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m.get(index) + scalar;
		++index;
	case 3:	out[index] = m.get(index) + scalar;
		++index;
	case 2:	out[index] = m.get(index) + scalar;
		++index;
	case 1:	out[index] = m.get(index) + scalar;
		++index;
		} while (index < s);
	}
	return out;
}
template<typename T>
Matrix<T> operator+(const T scalar, const Matrix<T> & m) {
	Matrix<T> out(m.rows(), m.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m.get(index) + scalar;
		++index;
	case 3:	out[index] = m.get(index) + scalar;
		++index;
	case 2:	out[index] = m.get(index) + scalar;
		++index;
	case 1:	out[index] = m.get(index) + scalar;
		++index;
		} while (index < s);
	}
	return out;
}

template<typename T>
Vector<T> operator*(const Matrix<T> & m, const Vector<T> & v) {
//	assert(m.cols() == v.size() && "Size mismatch");
	if (m.cols() != v.size()) throw UndefinedException("Size mismatch");
	Vector<T> out(m.rows());
	for (size_t i = 0; i < out.size(); ++i) {
		out[i] = 0;
		size_t index = 0;
		switch (m.cols() % 4) {
		case 0: do {
				out[i] += m.get(i, index) * v.get(index);
			++index;
		case 3:	out[i] += m.get(i, index) * v.get(index);
			++index;
		case 2: out[i] += m.get(i, index) * v.get(index);
			++index;
		case 1:	out[i] += m.get(i, index) * v.get(index);
			++index;
			} while (index < m.cols());
		}
	}
	return out;
}
#define min(a, b) ((a) < (b) ? (a) : (b))
template<typename T>
Matrix<T> operator*(const Matrix<T> & m1, const Matrix<T> & m2) {
//	assert(m1.cols() == m2.rows() && "Size mismatch");
	if (m1.cols() != m2.rows()) throw UndefinedException("Dimension mismatch");
	Matrix<T> out(m1.rows(), m2.cols());
	size_t stride = sqrt(32000 / sizeof(T) / 3);
	size_t rows = out.rows();
	for (size_t jj = 0; jj < out.cols(); jj += stride) {
		for (size_t kk = 0; kk < m1.cols(); kk += stride) {

			for (size_t i = 0; i < out.rows(); i += 2) {
				for (size_t j = jj; j < min(jj + stride - 1, out.cols()); j += 2) {
					for (size_t k = kk; k < min(kk + stride - 1, m1.cols()); ++k) {
						out(i, j) += m1.get(i, k) * m2.get(k, j);
						if(i + 1 < out.rows()) out(i + 1, j) += m1.get(i + 1, k) * m2.get(k, j);
						if(j + 1 < out.cols()) out(i, j + 1) += m1.get(i, k) * m2.get(k, j + 1);
						if(i + 1 < out.rows() && j + 1 < out.cols()) 
							out(i + 1, j + 1) += m1.get(i + 1, k) * m2.get(k, j + 1);
					}
				}
			}
		}
	}
	return out;
}

template<typename T>
Matrix<T> operator+(const Matrix<T> & m1, const Matrix<T> & m2) {
	if(m1.rows() != m2.rows() && m1.cols() != m2.cols()) throw UndefinedException("Dimension mismatch");
	Matrix<T> out(m1.rows(), m2.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m1.get(index) + m2.get(index);
		++index;
	case 3:	out[index] = m1.get(index) + m2.get(index);
		++index;
	case 2:	out[index] = m1.get(index) + m2.get(index);
		++index;
	case 1:	out[index] = m1.get(index) + m2.get(index);
		++index;
		} while (index < s);
	}
	return out;
}

//Element-wise matrix multiplication
template<typename T>
Matrix<T> hadamard(const Matrix<T> & m1, const Matrix<T> & m2) {
	if(m1.rows() != m2.rows() && m1.cols() != m2.cols()) throw UndefinedException("Dimension mismatch");
	Matrix<T> out(m1.rows(), m2.cols());
	size_t index = 0;
	size_t s = out.size();
	switch (s % 4) {
	case 0: do {
			out[index] = m1.get(index) * m2.get(index);
		++index;
	case 3:	out[index] = m1.get(index) * m2.get(index);
		++index;
	case 2:	out[index] = m1.get(index) * m2.get(index);
		++index;
	case 1:	out[index] = m1.get(index) * m2.get(index);
		++index;
	} while (index < s);
	}
	return out;
}
template<typename T>
std::ostream & operator<<(std::ostream & stream, const Matrix<T> & mat) {
	for (size_t i = 0; i < mat.rows(); ++i) {
		for (size_t j = 0; j < mat.cols(); ++j) {
			stream << mat.get(i, j) << " ";
		}
		stream << std::endl;
	}
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream & stream, const Matrix<T> && mat) {
	for (size_t i = 0; i < mat.rows(); ++i) {
		for (size_t j = 0; j < mat.cols(); ++j) {
			stream << mat.get(i, j) << " ";
		}
		stream << std::endl;
	}
	return stream;
}

template<typename T>
Matrix<T> transpose(const Vector<T> & v) {
	Matrix<T> mat(1, v.size());
	size_t index = 0;
	switch (mat.size() % 4) {
	case 0: do {
			mat[index] = v.get(index);
		++index;
	case 3:	mat[index] = v.get(index);
		++index;
	case 2:	mat[index] = v.get(index);
		++index;
	case 1:	mat[index] = v.get(index);
		++index;
		} while (index < mat.size());
	}
	return mat;
}

template<typename T>
Matrix<T> transpose(const Matrix<T> & m) {
	Matrix<T> mat(m.cols(), m.rows());
	size_t index = 0;
	switch (mat.size() % 4) {
	case 0: do {
		mat[index] = m.get(index % m.rows(), index / m.rows());
		++index;
	case 3:	mat[index] = m.get(index % m.rows(), index / m.rows());
		++index;
	case 2:	mat[index] = m.get(index % m.rows(), index / m.rows());
		++index;
	case 1:	mat[index] = m.get(index % m.rows(), index / m.rows());
		++index;
	} while (index < mat.size());
	}
	return mat;
}

using Mat = Matrix<double>;
using Matf = Matrix<float>;
using Mati = Matrix<int>;

//Fills the matrix  m with random numbers ranging from 0 to 1
const static void randomize(Matrix<> & m) {
	for (size_t i = 0; i < m.size(); ++i)
		m[i] = (double)rand() / RAND_MAX;
}

//Kernel convolution where the matrix is zero padded correctly so the kernel's bottom right element corresponds to the matrix's top left element
template<typename T>
Matrix<T> fullKernelConvolution(const Matrix<T> & kernel, const Matrix<T> & mat) {
	return kernel.applyAsKernel(mat.zeroPad(kernel.cols() - 1), 1);
}

template<typename T>
Matrix<T> removePadding(const Matrix<T> & mat, size_t padding) {
	if (padding == 0) return mat;
	if (padding >= mat.cols() || padding >= mat.rows()) throw UndefinedException(ERR_STR("Invalid padding"));
	Matrix<T> out(mat.rows() - 2 * padding, mat.cols() - 2 * padding);
	for (size_t x = padding; x < mat.cols() - padding; ++x) {
		for (size_t y = padding; y < mat.rows() - padding; ++y) {
			out(y - padding, x - padding) = mat.get(y, x);
		}
	}
	return out;
}