#pragma once
#include <vector>
#include "Error.h"
template<typename T = double>
class Vector
{
	class iterator {
		friend Vector;
	private:
		size_t pos;
		std::vector<T> & vec;
	public:
		iterator(size_t pos, std::vector<T> & data) : pos(pos), vec(data) {};
		T& operator*() {
			return vec[pos];
		}
		T operator*() const {
			return vec[pos];
		}
		T& operator->() {
			return vec[pos];
		}
		bool operator==(const iterator & other) const {
			return pos == other.pos;
		}
		bool operator!=(const iterator & other) const {
			return pos != other.pos;
		}
		bool operator<(const iterator & other) const {
			return pos < other.pos;
		}
		bool operator>(const iterator & other) const {
			return pos > other.pos;
		}
		bool operator<=(const iterator & other) const {
			return pos <= other.pos;
		}
		bool operator>=(const iterator & other) const {
			return pos >= other.pos;
		}
		iterator& operator++() {
			++pos;
			return *this;
		}
		iterator& operator++(int) {
			iterator cpy = *this;
			++pos;
			return cpy;
		}
		iterator& operator--() {
			--pos;
			return *this;
		}
		iterator& operator--(int) {
			iterator cpy = *this;
			--pos;
			return cpy;
		}
		friend iterator operator+(const iterator & it, size_t n);
		friend iterator operator+(size_t n, const iterator & it);
		friend iterator operator-(const iterator & it, size_t n);
		friend iterator operator-(size_t n, const iterator & it);
	};
private:
	std::vector<T> vec;
private:
public:
	Vector(size_t s) {
		vec.resize(s);
	}
	Vector(std::initializer_list<T> list) {
		vec.insert(vec.end(), list.begin(), list.end());
	}
	Vector(std::vector<T> & list) {
		vec.insert(vec.end(), list.begin(), list.end());
	}
	Vector(const Vector<T> & other) {
		vec = other.vec;
	}
	Vector(const iterator && start, const iterator && end) {
		vec.insert(vec.end(), start.vec.begin() + start.pos, end.vec.begin() + end.pos);
	}
	T& operator[](size_t index) throw(UndefinedException) {
		if (index >= vec.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return vec[index];
	}
	Vector<T> & operator=(const Vector<T> & other) noexcept {
		vec = other.vec;
		return *this;
	}
	explicit operator std::vector<T>() noexcept {
		return vec;
	}
	Vector<T> & operator*=(T n) noexcept {
		size_t index = 0;
		switch (vec.size() % 4) {
		case 0: do {
				vec[index++] *= n;
		case 3:	vec[index++] *= n;
		case 2:	vec[index++] *= n;
		case 1:	vec[index++] *= n;
			} while (index < vec.size());
		}
		return *this;
	}
	Vector<T> & operator+=(T n) noexcept {
		size_t index = 0;
		switch (vec.size() % 4) {
		case 0: do {
				vec[index++] += n;
		case 3:	vec[index++] += n;
		case 2:	vec[index++] += n;
		case 1:	vec[index++] += n;
			} while (index < vec.size());
		}
		return *this;
	}
	Vector<T> & operator+=(const Vector<T> & other) throw(UndefinedException) {
		if (vec.size() != other.size()) throw UndefinedException(ERR_STR("Size mismatch"));
		size_t index = 0;
		switch (vec.size() % 4) {
		case 0: do {
				vec[index] += other.get(index++);
		case 3:	vec[index] += other.get(index++);
		case 2:	vec[index] += other.get(index++);
		case 1:	vec[index] += other.get(index++);
			} while (index < vec.size());
		}
		return *this;
	}
	Vector<T> & operator-=(const Vector<T> & other) throw(UndefinedException) {
		if (vec.size() != other.size()) throw UndefinedException(ERR_STR("Size mismatch"));
		size_t index = 0;
		switch (vec.size() % 4) {
		case 0: do {
				vec[index] -= other.get(index++);
		case 3:	vec[index] -= other.get(index++);
		case 2:	vec[index] -= other.get(index++);
		case 1:	vec[index] -= other.get(index++);
			} while (index < vec.size());
		}
		return *this;
	}
	T magnitude() const {
		T mag;
		size_t index = 0;
		switch (vec.size() % 4) {
		case 0: do {
				mag += vec[index] * vec[index];
				++index;
		case 3:	mag += vec[index] * vec[index];
			++index;
		case 2:	mag += vec[index] * vec[index];
			++index;
		case 1:	mag += vec[index] * vec[index];
			++index;
			} while (index < vec.size());
		}
		return sqrt(mag);
	}
	Vector<T> unit() {
		size_t index = 0;
		Vector<T> v(vec.size());
		T mag = magnitude();
		switch (v.size() % 4) {
		case 0: do {
				v[index] = vec[index] / mag;
				++index;
		case 3: v[index] = vec[index] / mag;
			++index;
		case 2:	v[index] = vec[index] / mag;
			++index;
		case 1:	v[index] = vec[index] / mag;
			++index;
			} while (index < v.size());
		}
		return v;
	}
	void resize(size_t s) noexcept {
		vec.resize(s);
	}
	void zero() noexcept {
		std::fill(vec.begin(), vec.end(), 0);
	}
	size_t size() const noexcept {
		return vec.size();
	}
	T get(size_t index) const throw(UndefinedException) {
		if (index >= vec.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return vec[index];
	}
	iterator begin() noexcept {
		return iterator(0, vec);
	}
	iterator end() noexcept {
		return iterator(vec.size(), vec);
	}
	iterator at(size_t index) throw(UndefinedException) {
		if (index >= vec.size()) throw UndefinedException(ERR_STR("Index out of bounds"));
		return iterator(index, vec);
	}
	decltype(auto) vbegin() noexcept {
		return vec.begin();
	}
	decltype(auto) vend() noexcept {
		return vec.end();
	}
	Vector() = default;
	~Vector() = default;
	inline void insert(const iterator && position, const iterator && start, const iterator && end) {
		vec.insert(vec.begin() + position.pos, start.vec.begin() + start.pos, end.vec.begin() + end.pos);
	}
};

template<typename T>
Vector<T> operator*(const Vector<T> & v, const T n) {
	Vector<T> vec(v.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v.get(index) * n;
			++index;
	case 3:	vec[index] = v.get(index) * n;
		++index;
	case 2:	vec[index] = v.get(index) * n;
		++index;
	case 1:	vec[index] = v.get(index) * n;
		++index;
		} while (index < vec.size());
	}
	return vec;
}
template<typename T>
Vector<T> operator*(const T n, const Vector<T> & v) {
	Vector<T> vec(v.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v.get(index) * n;
			++index;
	case 3:	vec[index] = v.get(index) * n;
		++index;
	case 2:	vec[index] = v.get(index) * n;
		++index;
	case 1:	vec[index] = v.get(index) * n;
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
Vector<T> operator+(const T n, const Vector<T> & v) {
	Vector<T> vec(v.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v.get(index) + n;
			++index;
	case 3:	vec[index] = v.get(index) + n;
		++index;
	case 2:	vec[index] = v.get(index) + n;
		++index;
	case 1:	vec[index] = v.get(index) + n;
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
Vector<T> operator+(const Vector<T> & v, const T n) {
	Vector<T> vec(v.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v.get(index) + n;
			++index;
	case 3:	vec[index] = v.get(index) + n;
		++index;
	case 2:	vec[index] = v.get(index) + n;
		++index;
	case 1:	vec[index] = v.get(index) + n;
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
Vector<T> operator+(const Vector<T> & v1, const Vector<T> & v2) throw(UndefinedException) {
	if (v1.size() != v2.size()) throw UndefinedException(ERR_STR("Sizes must match"));
	Vector<T> vec(v1.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v1.get(index) + v2.get(index);
			++index;
	case 3:	vec[index] = v1.get(index) + v2.get(index);
		++index;
	case 2:	vec[index] = v1.get(index) + v2.get(index);
		++index;
	case 1:	vec[index] = v1.get(index) + v2.get(index);
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
Vector<T> operator-(const Vector<T> & v1, const Vector<T> & v2) throw(UndefinedException) {
	if (v1.size() != v2.size()) throw UndefinedException(ERR_STR("Sizes must match"));
	Vector<T> vec(v1.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v1.get(index) - v2.get(index);
			++index;
	case 3:	vec[index] = v1.get(index) - v2.get(index);
		++index;
	case 2:	vec[index] = v1.get(index) - v2.get(index);
		++index;
	case 1:	vec[index] = v1.get(index) - v2.get(index);
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
Vector<T> hadamard(const Vector<T> & v1, const Vector<T> & v2) throw (UndefinedException) {
	if (v1.size() != v2.size()) throw UndefinedException(ERR_STR("Sizes must match"));
	Vector<T> vec(v1.size());
	size_t index = 0;
	switch (vec.size() % 4) {
	case 0: do {
			vec[index] = v1.get(index) * v2.get(index);
			++index;
	case 3:	vec[index] = v1.get(index) * v2.get(index);
		++index;
	case 2:	vec[index] = v1.get(index) * v2.get(index);
		++index;
	case 1:	vec[index] = v1.get(index) * v2.get(index);
		++index;
		} while (index < vec.size());
	}
	return vec;
}

template<typename T>
T dot(const Vector<T> & v1, const Vector<T> & v2) {
	if (v1.size() != v2.size()) throw UndefinedException(ERR_STR("Sizes must match"));
	T dt = 0;
	size_t index = 0;
	switch (v1.size() % 4) {
	case 0: do {
			dt += v1.get(index) * v2.get(index);
			++index;
	case 3:	dt += v1.get(index) * v2.get(index);
		++index;
	case 2:	dt += v1.get(index) * v2.get(index);
		++index;
	case 1:	dt += v1.get(index) * v2.get(index);
		++index;
		} while (index < v1.size());
	}
	return dt;
}

template<typename T>
T dist(const Vector<T> & v1, const Vector<T> & v2) {
	if (v1.size() != v2.size()) throw UndefinedException(ERR_STR("Sizes must match"));
	T dist = 0;
	size_t index = 0;
	switch (v1.size() % 4) {
	case 0: do {
			dist += (v1.get(index) - v2.get(index)) * (v1.get(index) - v2.get(index));
		++index;
	case 3:	dist += (v1.get(index) - v2.get(index)) * (v1.get(index) - v2.get(index));
		++index;
	case 2:	dist += (v1.get(index) - v2.get(index)) * (v1.get(index) - v2.get(index));
		++index;
	case 1:	dist += (v1.get(index) - v2.get(index)) * (v1.get(index) - v2.get(index));
		++index;
		} while (index < v1.size());
	}
	return sqrt(dist);
}

template<typename T>
std::ostream & operator<<(std::ostream & stream, Vector<T> & vec) {
	for (auto e : vec)
		stream << e << "\n";
	return stream;
}

template<typename T>
std::ostream & operator<<(std::ostream & stream, Vector<T> && vec) {
	for (auto e : vec)
		stream << e << "\n";
	return stream;
}

template<typename T = double> 
typename Vector<T>::iterator operator+(const typename Vector<T>::iterator & it, size_t n) {
	return typename Vector<T>::iterator(it.pos + n);
}
template<typename T = double>
typename Vector<T>::iterator operator+(size_t n, const typename Vector<T>::iterator & it) {
	return typename Vector<T>::iterator(it.pos + n);
}
template<typename T = double>
typename Vector<T>::iterator operator-(const typename Vector<T>::iterator & it, size_t n) {
	return typename Vector<T>::iterator(it.pos - n);
}
template<typename T = double>
typename Vector<T>::iterator operator-(size_t n, const typename Vector<T>::iterator & it) {
	return typename Vector<T>::iterator(it.pos - n);
}
const static void randomize(Vector<> & v) {
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = (double)rand() / RAND_MAX;
}
const static void randomize(std::vector<double> & v) {
	for (size_t i = 0; i < v.size(); ++i)
		v[i] = (double)rand() / RAND_MAX;
}
const static Vector<> mean(std::vector<Vector<>> & v) {
	Vector<> out = v[0];
	auto end = v.end();
	for (auto it = v.begin() + 1; it != end; ++it)
		out += *it;
	out *= 1.0 / v.size();
	return out;
}
//Returns the index of the maximum element
//For use in converting output of fc layer to categories
template <typename T>
const static size_t getMaxIndex(const Vector<T> & v) {
	T max = std::numeric_limits<T>::min();
	size_t index;
	for (size_t i = 0; i < v.size(); ++i) {
		if (v.get(i) > max) {
			index = i;
			max = v.get(i);
		}
	}
	return index;
}