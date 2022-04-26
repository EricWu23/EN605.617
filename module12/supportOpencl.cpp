
#include <iostream>
#include <fstream>
#include <sstream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "assignment.h"
#include "supportOpencl.h"

// Function to check and handle OpenCL errors
void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
cl_int GetNumOfPlatforms(cl_uint* numPlatforms){
    cl_int errNum;
    errNum = clGetPlatformIDs(0, NULL, numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (*numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs");
    std::cout << "Number of platforms: \t" << *numPlatforms << std::endl; 
}

cl_int GetPlatformIDs(cl_uint numPlatforms,cl_platform_id * platformIDs){
    cl_int errNum;

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");
    return errNum;
}

cl_int GetNumOfAvaliableDevices(cl_uint* numDevices,cl_platform_id platform,cl_device_type device_type){
  cl_int errNum;
  errNum = clGetDeviceIDs(
        platform, 
        device_type, 
        0,
        NULL,
        numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }     
}

cl_int GetDeviceIDs(cl_platform_id platform,cl_device_type device_type,cl_uint num_entry,cl_device_id *devices,cl_uint *num_devices){
    cl_int errNum;
    errNum = clGetDeviceIDs(
        platform,
        device_type,
        num_entry, 
        devices, 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");
}

cl_context CreateContext(cl_platform_id PlatformId,cl_uint num_devices,const cl_device_id *devices)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_context context = NULL;

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)PlatformId,
        0
    };
    context = clCreateContext(
        contextProperties, 
        num_devices,
        devices, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    return context;
}


cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        NULL, NULL);
    if (program == NULL)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device)
{
    cl_int errNum;
    cl_command_queue commandQueue = NULL;

	commandQueue = clCreateCommandQueue(
		context,
		device,
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");
    return commandQueue;
}