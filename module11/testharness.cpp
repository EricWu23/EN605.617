
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "testharness.h"
//#include "supportOpencl.h"
#include "assignment.h"
#include "utility.h"

#if !defined(CL_CALLBACK)
#define CL_CALLBACK
#endif

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}


void CL_CALLBACK contextCallback(
	const char * errInfo,
	const void * private_info,
	size_t cb,
	void * user_data)
{
	std::cout << "Error occured during context use: " << errInfo << std::endl;
	// should really perform any clearup and so on at this point
	// but for simplicitly just exit.
	exit(1);
}


static int testconvolve(cl_uint* inputSignal,cl_uint* mask, cl_uint* outputSignal,
unsigned int inputSignalWidth,unsigned int inputSignalHeight,
unsigned int maskWidth,unsigned int maskHeight,
unsigned int outputSignalWidth,unsigned int outputSignalHeight){


    cl_int errNum;
    cl_uint numPlatforms;
	cl_uint numDevices;
    cl_platform_id * platformIDs;
	cl_device_id * deviceIDs;
    cl_context context = NULL;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem inputSignalBuffer;
	cl_mem outputSignalBuffer;
	cl_mem maskBuffer;

    // 1. Get a list of available OpenCL Platforms  
	errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr( 
		(errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
		"clGetPlatformIDs"); 
 
	platformIDs = (cl_platform_id *)alloca(
       		sizeof(cl_platform_id) * numPlatforms);

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
	   (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
	   "clGetPlatformIDs");

	// 2. For each platform, get the avalaible GPU devices
	deviceIDs = NULL;
	cl_uint i;
	for (i = 0; i < numPlatforms; i++)
	{
		errNum = clGetDeviceIDs(
            platformIDs[i], 
            CL_DEVICE_TYPE_GPU, 
            0,
            NULL,
            &numDevices);
		if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
	    {
			checkErr(errNum, "clGetDeviceIDs");
        }
	    else if (numDevices > 0) 
	    {
		   	deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
			errNum = clGetDeviceIDs(
				platformIDs[i],
				CL_DEVICE_TYPE_GPU,
				numDevices, 
				&deviceIDs[0], 
				NULL);
			checkErr(errNum, "clGetDeviceIDs");
			break;
	   }
	}

      // 3. create an OpenCL context on the selected platform (the first avaliable platform).  
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[0],
        0
    };
    context = clCreateContext(
		contextProperties, 
		numDevices,
        deviceIDs, 
		&contextCallback,
		NULL, 
		&errNum);
	checkErr(errNum, "clCreateContext");
    
    //4. Create program object from source kernel file
    std::ifstream srcFile("Convolution.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading Convolution.cl");

	std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

	const char * src = srcProg.c_str();
	size_t length = srcProg.length();

	program = clCreateProgramWithSource(
		context, 
		1, 
		&src, 
		&length, 
		&errNum);
	checkErr(errNum, "clCreateProgramWithSource");

    //5. Build program object at run-time
	errNum = clBuildProgram(
		program,
		numDevices,
		deviceIDs,
		NULL,
		NULL,
		NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
			program, 
			deviceIDs[0], 
			CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
			buildLog, 
			NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
		checkErr(errNum, "clBuildProgram");
    }

    //6. Create kernel object using the program object
	kernel = clCreateKernel(
		program,
		"convolve",
		&errNum);
	checkErr(errNum, "clCreateKernel"); 

    //7. Create memory buffer objects
	inputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * inputSignalHeight * inputSignalWidth,
		static_cast<void *>(inputSignal),
		&errNum);
	checkErr(errNum, "clCreateBuffer(inputSignal)");

	maskBuffer = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint) * maskHeight * maskWidth,
		static_cast<void *>(mask),
		&errNum);
	checkErr(errNum, "clCreateBuffer(mask)");

	outputSignalBuffer = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * outputSignalHeight * outputSignalWidth,
		NULL,
		&errNum);
	checkErr(errNum, "clCreateBuffer(outputSignal)");

    // 8.Pick the first device and create command queue.
	queue = clCreateCommandQueue(
		context,
		deviceIDs[0],
		CL_QUEUE_PROFILING_ENABLE,
		&errNum);
	checkErr(errNum, "clCreateCommandQueue");

    //9. Set the kernel arguments
    errNum  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &maskBuffer);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &outputSignalBuffer);
	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &inputSignalWidth);
	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_uint), &maskWidth);
	checkErr(errNum, "clSetKernelArg");

    //10. Enque kernel execution command
    const size_t globalWorkSize[2] = { outputSignalWidth, outputSignalHeight };
    const size_t localWorkSize[2]  = { 1, 1 };

    // Queue the kernel up for execution across the array
    cl_event event;
    errNum = clEnqueueNDRangeKernel(
		queue, 
		kernel, 
		2,
		NULL,
        globalWorkSize, 
		localWorkSize,
        0, 
		NULL, 
		&event);
	checkErr(errNum, "clEnqueueNDRangeKernel");
    
    clWaitForEvents(1, &event);
    clFinish(queue);
    cl_ulong time_start; cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;
    std::cout << "Kernel execution "<<"convolve" <<" took "<<nanoSeconds/1000000.0<<" milli seconds."<<std::endl;

    //11. read the result  
    errNum = clEnqueueReadBuffer(
		queue, 
		outputSignalBuffer, 
		CL_TRUE,
        0, 
		sizeof(cl_uint) * outputSignalHeight * outputSignalHeight, 
		outputSignal,
        0, 
		NULL, 
		NULL);
	checkErr(errNum, "clEnqueueReadBuffer");

    // Output the result buffer
    for (int y = 0; y < outputSignalHeight; y++)
	{
		for (int x = 0; x < outputSignalWidth; x++)
		{
			std::cout << outputSignal[y*outputSignalWidth+x] << " ";
		}
		std::cout << std::endl;
	}

    std::cout << std::endl << "Executed program succesfully." << std::endl;

	return 0;
}

void testHarness(){

    unsigned int padding=0;
    unsigned int stride=1;
    // init a 2-D input array
    const unsigned int inputSignalWidth  = 49;
    const unsigned int inputSignalHeight = 49;
    cl_uint inputSignal[inputSignalHeight][inputSignalWidth]={0};
    // cl_uint inputSignal[inputSignalHeight][inputSignalWidth] =
    // {
    //     {3, 1, 1, 4, 8, 2, 1, 3},
    //     {4, 2, 1, 1, 2, 1, 2, 3},
    //     {4, 4, 4, 4, 3, 2, 2, 2},
    //     {9, 8, 3, 8, 9, 0, 0, 0},
    //     {9, 3, 3, 9, 0, 0, 0, 0},
    //     {0, 9, 0, 8, 0, 0, 0, 0},
    //     {3, 0, 8, 8, 9, 4, 4, 4},
    //     {5, 9, 8, 1, 8, 1, 1, 1}
    // };
    cpu_array0_int<cl_uint>(&inputSignal[0][0],inputSignalHeight,inputSignalWidth);

    // init a 2-D mask array
    const unsigned int maskWidth  = 7;
    const unsigned int maskHeight = 7;
     cl_uint mask[maskHeight][maskWidth] =
     {
     	{25,25,25,25,25,25,25},
        {25,50,50,50,50,50,25},
        {25,50,75,75,75,50,25},
        {25,50,75,100,75,50,25},
        {25,50,75,75,75,50,25},
        {25,50,50,50,50,50,25},
        {25,25,25,25,25,25,25}
     };
    // init a 2-D output array
     const unsigned int outputSignalWidth  = (inputSignalWidth+2*padding-maskWidth)/stride+1;
     const unsigned int outputSignalHeight = (inputSignalHeight+2*padding-maskHeight)/stride+1;

     cl_uint outputSignal[outputSignalHeight][outputSignalWidth];
    // call the testconvolve
    
    testconvolve(&inputSignal[0][0],&mask[0][0],&outputSignal[0][0],
                 inputSignalWidth,inputSignalHeight,
                 maskWidth,maskHeight,
                 outputSignalWidth,outputSignalHeight);

}