
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

template<typename T>
void appendBitfield(T info, T value, std::string name, std::string & str)
{
	if (info & value) 
	{
		if (str.length() > 0)
		{
			str.append(" | ");
		}
		str.append(name);
	}
}	

///
// Display information for a particular device.
// As different calls to clGetDeviceInfo may return
// values of different types a template is used. 
// As some values returned are arrays of values, a templated class is
// used so it can be specialized for this case, see below.
//
template <typename T>
class InfoDevice
{
public:
	static void display(
		cl_device_id id, 
		cl_device_info name,
		std::string str)
	{
		cl_int errNum;
		std::size_t paramValueSize;

		errNum = clGetDeviceInfo(
			id,
			name,
			0,
			NULL,
			&paramValueSize);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		T * info = (T *)alloca(sizeof(T) * paramValueSize);
		errNum = clGetDeviceInfo(
			id,
			name,
			paramValueSize,
			info,
			NULL);
		if (errNum != CL_SUCCESS)
		{
			std::cerr << "Failed to find OpenCL device info " << str << "." << std::endl;
			return;
		}

		// Handle a few special cases
		switch (name)
		{
		case CL_DEVICE_TYPE:
			{
				std::string deviceType;

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_CPU, 
					"CL_DEVICE_TYPE_CPU", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_GPU, 
					"CL_DEVICE_TYPE_GPU", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_ACCELERATOR, 
					"CL_DEVICE_TYPE_ACCELERATOR", 
					deviceType);

				appendBitfield<cl_device_type>(
					*(reinterpret_cast<cl_device_type*>(info)),
					CL_DEVICE_TYPE_DEFAULT, 
					"CL_DEVICE_TYPE_DEFAULT", 
					deviceType);

				std::cout << "\t\t" << str << ":\t" << deviceType << std::endl;
			}
			break;
		case CL_DEVICE_SINGLE_FP_CONFIG:
			{
				std::string fpType;
				
				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_DENORM, 
					"CL_FP_DENORM", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_INF_NAN, 
					"CL_FP_INF_NAN", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_NEAREST, 
					"CL_FP_ROUND_TO_NEAREST", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_ZERO, 
					"CL_FP_ROUND_TO_ZERO", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_ROUND_TO_INF, 
					"CL_FP_ROUND_TO_INF", 
					fpType); 

				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_FMA, 
					"CL_FP_FMA", 
					fpType); 

#ifdef CL_FP_SOFT_FLOAT
				appendBitfield<cl_device_fp_config>(
					*(reinterpret_cast<cl_device_fp_config*>(info)),
					CL_FP_SOFT_FLOAT, 
					"CL_FP_SOFT_FLOAT", 
					fpType); 
#endif

				std::cout << "\t\t" << str << ":\t" << fpType << std::endl;
			}
		case CL_DEVICE_GLOBAL_MEM_CACHE_TYPE:
			{
				std::string memType;
				
				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_NONE, 
					"CL_NONE", 
					memType); 
				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_READ_ONLY_CACHE, 
					"CL_READ_ONLY_CACHE", 
					memType); 

				appendBitfield<cl_device_mem_cache_type>(
					*(reinterpret_cast<cl_device_mem_cache_type*>(info)), 
					CL_READ_WRITE_CACHE, 
					"CL_READ_WRITE_CACHE", 
					memType); 

				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_LOCAL_MEM_TYPE:
			{
				std::string memType;
				
				appendBitfield<cl_device_local_mem_type>(
					*(reinterpret_cast<cl_device_local_mem_type*>(info)), 
					CL_GLOBAL, 
					"CL_LOCAL", 
					memType);

				appendBitfield<cl_device_local_mem_type>(
					*(reinterpret_cast<cl_device_local_mem_type*>(info)), 
					CL_GLOBAL, 
					"CL_GLOBAL", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_EXECUTION_CAPABILITIES:
			{
				std::string memType;
				
				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_EXEC_KERNEL, 
					"CL_EXEC_KERNEL", 
					memType);

				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_EXEC_NATIVE_KERNEL, 
					"CL_EXEC_NATIVE_KERNEL", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		case CL_DEVICE_QUEUE_PROPERTIES:
			{
				std::string memType;
				
				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 
					"CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", 
					memType);

				appendBitfield<cl_device_exec_capabilities>(
					*(reinterpret_cast<cl_device_exec_capabilities*>(info)), 
					CL_QUEUE_PROFILING_ENABLE, 
					"CL_QUEUE_PROFILING_ENABLE", 
					memType);
				
				std::cout << "\t\t" << str << ":\t" << memType << std::endl;
			}
			break;
		default:
			std::cout << "\t\t" << str << ":\t" << *info << std::endl;
			break;
		}
	}
};

cl_context CreateContext()
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;

    // First, select an OpenCL platform to run on.  For this example, we
    // simply choose the first available platform.  Normally, you would
    // query for all available platforms and select the most appropriate one.
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    // Next, create an OpenCL context on the platform.  Attempt to
    // create a GPU-based context, and if that fails, try to create
    // a CPU-based context.
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,
                                      NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS)
    {
        std::cout << "Could not create GPU context, trying CPU..." << std::endl;
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          NULL, NULL, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
            return NULL;
        }
    }

    return context;
}


cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device,cl_command_queue_properties properties)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = NULL;
    size_t deviceBufferSize = -1;

    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return NULL;
    }

    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return NULL;
    }

    // Allocate memory for the devices buffer
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return NULL;
    }

    // In this example, we just choose the first available device.  In a
    // real program, you would likely use all available devices or choose
    // the highest performance device based on OpenCL device queries
    commandQueue = clCreateCommandQueue(context, devices[0],properties, NULL);
    if (commandQueue == NULL)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return NULL;
    }
/*
    InfoDevice<cl_device_type>::display(
				devices[0], 
				CL_DEVICE_TYPE, 
				"CL_DEVICE_TYPE");
    
    InfoDevice<cl_uint>::display(
        devices[0], 
        CL_DEVICE_VENDOR_ID, 
        "CL_DEVICE_VENDOR_ID");
  */
    *device = devices[0];
    delete [] devices;
    return commandQueue;
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


bool CreateMemObjects(cl_context context, cl_mem memObjects[2],
                      float *a, float *b)
{
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, a, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * ARRAY_SIZE, b, NULL);


    if (memObjects[0] == NULL || memObjects[1] == NULL)
    {
        std::cerr << "Error creating memory objects." << std::endl;
        return false;
    }

    return true;
}


void Cleanup(cl_context context, cl_command_queue commandQueue,
             cl_program program, cl_kernel kernel, cl_mem memObjects[3])
{
    for (int i = 0; i < 3; i++)
    {
        if (memObjects[i] != 0)
            clReleaseMemObject(memObjects[i]);
    }
    if (commandQueue != 0)
        clReleaseCommandQueue(commandQueue);

    if (kernel != 0)
        clReleaseKernel(kernel);

    if (program != 0)
        clReleaseProgram(program);

    if (context != 0)
        clReleaseContext(context);

}

