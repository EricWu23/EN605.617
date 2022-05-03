//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// raytracer.cpp
//
//    This is a (very) simple raytracer that is intended to demonstrate 
//    using OpenCL buffers.

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "info.hpp"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context0;
    cl_program program0;
    int * inputOutput0;
    cl_context context1;
    cl_program program1;
    int * inputOutput1;

    int platform = DEFAULT_PLATFORM; 

    std::cout << "Simple buffer and sub-buffer Example" << std::endl;

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("simple.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading simple.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    // context0 groups all the devices avaliable on the platform defined in contextProperties
    context0 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // context1 groups all the devices avaliable on the platform defined in contextProperties
    context1 = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    // Create program object program0 from source and associates it with the context0
    program0 = clCreateProgramWithSource(
        context0, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

     // Create program object program1 from source and associates it with the context1
    program1 = clCreateProgramWithSource(
        context1, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    // Build program0 on all the devices
    errNum = clBuildProgram(
        program0,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
     // Build program1 on all the devices
    errNum |= clBuildProgram(
        program1,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);
 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog0[16384];
        clGetProgramBuildInfo(
            program0, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog0), 
            buildLog0, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog0;
            checkErr(errNum, "clBuildProgram");
     
        // Determine the reason for the error
        char buildLog1[16384];
        clGetProgramBuildInfo(
            program1, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog1), 
            buildLog1, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog1;
            checkErr(errNum, "clBuildProgram");
    }

    // create buffers and sub-buffers
    inputOutput0 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput0[i] = i;
    }
 
    inputOutput1 = new int[NUM_BUFFER_ELEMENTS * numDevices];
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS * numDevices; i++)
    {
        inputOutput1[i] = i;
    }

    // create a single buffer buffer0 on context0 to cover all the input data
    cl_mem buffer0 = clCreateBuffer(
        context0,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
 
     // create a single buffer buffer1 on context1 to cover all the input data
    cl_mem buffer1 = clCreateBuffer(
        context1,
        CL_MEM_READ_WRITE,
        sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");

    // Create command queues
    InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");

    // create commandqueue queue0 for the first device in context0
    cl_command_queue queue0 = 
     	clCreateCommandQueue(
     	context0,
     	deviceIDs[0],
     	0,//in-order queue
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");

    // create commandqueue queue1 for the first device in context1
    cl_command_queue queue1 = 
     	clCreateCommandQueue(
     	context1,
     	deviceIDs[0],
     	0,//in-order queue
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");
 
    //create kernel object kernel0 from program0 for kernel function square
    cl_kernel kernel0 = clCreateKernel(
     program0,
     "square",
     &errNum);
    checkErr(errNum, "clCreateKernel(square)");
 
     //create kernel object kernel1 from program1 for kernel function cube
    cl_kernel kernel1 = clCreateKernel(
     program1,
     "cube",
     &errNum);
    checkErr(errNum, "clCreateKernel(cube)");

    //define input argument for each kernel

    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), (void *)&buffer0);// kernel0 will square the data in buffer0
    checkErr(errNum, "clSetKernelArg(square)");

    errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), (void *)&buffer1);//kernel1 will cube the data in buffer1
    checkErr(errNum, "clSetKernelArg(cube)");
 
    // Write input data
    // write data from inputOutput0 to buffer 0
    errNum = clEnqueueWriteBuffer(
      queue0,//quueue0 needs to be long to context0 and it is
      buffer0,//buffer object created previously on context0
      CL_TRUE,// blocking write
      0,
      sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,//the size of bytes being write
      (void*)inputOutput0,// host buffer contains the data that will be written into the buffer0 on device
      0,
      NULL,
      NULL);
    
    // write data from inputOutput1 to buffer 1
    errNum = clEnqueueWriteBuffer(
      queue1,
      buffer1,
      CL_TRUE,
      0,
      sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
      (void*)inputOutput1,
      0,
      NULL,
      NULL);
 
    std::vector<cl_event> events;
    // call kernel for each device
    cl_event event0;

    size_t gWI = NUM_BUFFER_ELEMENTS;

    //enqueue the kernel0 into queue0 (in-order queue)
    errNum = clEnqueueNDRangeKernel(
      queue0, 
      kernel0,// kernel0 is the square kernel associated with context0 
      1, 
      NULL,
      (const size_t*)&gWI, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event0);
	
    //create a marker on queue1 and emit event1
 	  cl_event event1;
 	  errNum = clEnqueueMarker(queue1, &event1);

    //enqueue the kernel1 into queue1 (in-order queue)
    errNum = clEnqueueNDRangeKernel(
      queue1, 
      kernel1,//kernel1 is the cube kernel associated with context1
      1, 
      NULL,
      (const size_t*)&gWI, 
      (const size_t*)NULL, 
      0, 
      0, 
      &event0); 
 	
 	//Wait for queue 1 to complete before continuing on queue 0
 	errNum = clEnqueueBarrier(queue0);//create a barrier on queue0
 	errNum = clEnqueueWaitForEvents(queue0, 1, &event1);// wait for the event1 to preceed the commands on queue0.

 	// Read back computed data
   	clEnqueueReadBuffer(
            queue0,
            buffer0,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput0,//data readback from buffer0 to inputOutput0
            0,
            NULL,
            NULL);
   	clEnqueueReadBuffer(
            queue1,
            buffer1,
            CL_TRUE,
            0,
            sizeof(int) * NUM_BUFFER_ELEMENTS * numDevices,
            (void*)inputOutput1,//data readback from buffer1 to inputOutput1
            0,
            NULL,
            NULL);
 
    // Display output in rows
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
    {
     std::cout << " " << inputOutput0[elems];
    }
    std::cout << std::endl;
 
    for (unsigned elems = 0; elems < NUM_BUFFER_ELEMENTS; elems++)
    {
     std::cout << " " << inputOutput1[elems];
    }
    std::cout << std::endl;
 
    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
