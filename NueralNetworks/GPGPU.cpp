#include "GPGPU.h"
#ifdef _GPGPU
#include <fstream>
#include <sstream>


GPGPU::GPGPU() : deviceId(NULL), platformId(NULL)
{
	cl_int ret = clGetPlatformIDs(1, &platformId, &numPlatforms);
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceId, &numDevices);
	ctx = clCreateContext(NULL, 1, &deviceId, NULL, NULL, &ret);
	q = clCreateCommandQueue(ctx, deviceId, 0, &ret);
}

GPGPU::GPGPU(const char * filename) : GPGPU()
{
	loadSource(filename);
}


GPGPU::~GPGPU()
{
	clFlush(q);
	clFinish(q);
	for (auto & k : kernels)
		clReleaseKernel(k.second);
	clReleaseProgram(program);
	for (auto & a : args) {
		for (auto & v : a.second)
			clReleaseMemObject(v.data);
	}
	clReleaseCommandQueue(q);
	clReleaseContext(ctx);
}

void GPGPU::loadSource(const char * filename)
{
	std::ifstream in(filename);
	std::stringstream buffer;
	buffer << in.rdbuf();
	cl_int ret;
	const char * source = buffer.str().c_str();
	const size_t size = buffer.str().size();
	program = clCreateProgramWithSource(ctx, 1, &source, &size, &ret);
	ret = clBuildProgram(program, 1, &deviceId, NULL, NULL, NULL);
}

void GPGPU::createKernel(std::string function)
{
	cl_int ret;
	kernels[function] = clCreateKernel(program, function.c_str(), &ret);
}

void GPGPU::passArg(std::string kernel, cl_uint param, std::string paramName, void * data, size_t size, cl_mem_flags type)
{
	cl_int ret;
	cl_mem mem = clCreateBuffer(ctx, type, size, NULL, &ret);
	if (data != nullptr) {
		ret = clEnqueueWriteBuffer(q, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
	}
	args[kernel].emplace_back(mem, paramName);
	ret = clSetKernelArg(kernels[kernel], param, sizeof(cl_mem), &mem);

}

void GPGPU::exec(std::string kernel, size_t totalSize, size_t localSize)
{
	cl_int ret;
	ret = clEnqueueNDRangeKernel(q, kernels[kernel], 1, NULL, &totalSize, &localSize, 0, NULL, NULL);
}

void GPGPU::read(std::string kernel, std::string param, void * data, size_t size)
{
	auto & params = args[kernel];
	cl_mem mem;
	for (auto & v : params) {
		if (v.name == param) {
			mem = v.data;
			break;
		}
	}
	cl_int ret = clEnqueueReadBuffer(q, mem, CL_TRUE, 0, size, data, 0, NULL, NULL);
}
#endif

