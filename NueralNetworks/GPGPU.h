#pragma once
#ifdef _GPGPU
#include <CL\cl.h>
#include <unordered_map>
struct arg {
	void * data;
	size_t size;
};
struct var {
	cl_mem data;
	std::string name;
	var(cl_mem d, std::string n) : data(d), name(n) {};
	var() = default;
};
class GPGPU
{
private:
	cl_context ctx;
	cl_command_queue q;
	cl_platform_id platformId;
	cl_device_id deviceId;
	cl_uint numPlatforms, numDevices;
	std::unordered_map<std::string, cl_kernel> kernels;
	std::unordered_map<std::string, std::vector<var>> args;
	cl_program program;
public:
	GPGPU();
	GPGPU(const char * filename);
	~GPGPU();
	void loadSource(const char * filename);
	void createKernel(std::string function);
	void passArg(std::string kernel, cl_uint param, std::string paramName, void * data, size_t size, cl_mem_flags type);
	void exec(std::string kernel, size_t totalSize, size_t localSize);
	void read(std::string kernel, std::string param, void * data, size_t size);
}; 
#endif
