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
#define SUB_BUFFER_WIDTH 2
#define SUB_BUFFER_HEIGHT 2

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
    cl_context context;
    cl_program program;
    std::vector<cl_mem> buffers;//vector storing the subbuffers for input buffer
    std::vector<cl_mem> outputbuffers;//vector storing the subbuffers for output buffer
    cl_uint subbuffer_width=SUB_BUFFER_WIDTH;
    cl_uint subbuffer_height=SUB_BUFFER_HEIGHT;
    int * inputOutput;

    int platform = DEFAULT_PLATFORM; 


    std::cout << "Assignment 12: 2x2 averagebuffer as moving average" << std::endl;


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

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");

    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");

    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
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

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }
        // Create command queues use the first avaliable device
    InfoDevice<cl_device_type>::display(
        deviceIDs[0], 
        CL_DEVICE_TYPE, 
        "CL_DEVICE_TYPE");

    cl_command_queue queue = 
        clCreateCommandQueue(
            context,
            deviceIDs[0],
            0,
            &errNum);
    checkErr(errNum, "clCreateCommandQueue");

    
    // create a single buffer to cover all the input data, data will be filled later
    // for our purpose, we have to pad subbuffer_width*subbuffer_height-1 zeros
    uint elementsafterpadding=NUM_BUFFER_ELEMENTS+SUB_BUFFER_WIDTH*SUB_BUFFER_HEIGHT-1;
    
    // create buffers and sub-buffers
    inputOutput = new int[elementsafterpadding];//host
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    { 
        inputOutput[i] = (float)i;
    }
    //zero padding
    for(unsigned int i=NUM_BUFFER_ELEMENTS;i<elementsafterpadding;i++){
        inputOutput[i] = (float)0.0;
    }

    cl_mem main_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * elementsafterpadding,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    // Write input data (copy data from inputOutput into main_buffer)
    errNum = clEnqueueWriteBuffer(
        queue,
        main_buffer,
        CL_TRUE,
        0,
        sizeof(int) * elementsafterpadding,
        (void*)inputOutput,
        0,
        NULL,
        NULL);
    // create a 2x2 sub-buffer for each element in the original input buffer
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {

        cl_buffer_region region = 
            {
                i * sizeof(float),//offset in bytes of subbuffer in the buffer
                SUB_BUFFER_WIDTH*SUB_BUFFER_HEIGHT * sizeof(float)//total bytes of sub-buffer
            };
        cl_mem buffer = clCreateSubBuffer(
            main_buffer,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        buffers.push_back(buffer);
    }

    // create a single buffer to store the computed output
    cl_mem out_buffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        NULL,
        &errNum);
    checkErr(errNum, "clCreateBuffer");
    // create subbuffer from the output buffer
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {

        cl_buffer_region region = 
            {
                i * sizeof(float),//offset in bytes of subbuffer in the buffer
                sizeof(float)//total bytes of sub-buffer
            };
        cl_mem buffer = clCreateSubBuffer(
            out_buffer,
            CL_MEM_WRITE_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        outputbuffers.push_back(buffer);
    }


    cl_kernel kernel = clCreateKernel(
        program,
        "averagebuffer",
        &errNum);
    checkErr(errNum, "clCreateKernel(averagebuffer)");
    
    std::vector<cl_event> events;
    size_t gWI[2] = {subbuffer_width,subbuffer_height};
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
      errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
      errNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &subbuffer_width);
      errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &subbuffer_height);
      errNum |= clSetKernelArg(kernel, 3, sizeof(cl_uint), &outputbuffers[i]);
      checkErr(errNum, "clSetKernelArg(averagebuffer)");
      // call kernel for each subbuffer
      cl_event event;
      errNum = clEnqueueNDRangeKernel(
          queue, 
          kernel, 
          2, 
          NULL,
          (const size_t*)&gWI, 
          (const size_t*)NULL, 
          0, 
          0, 
          &event);
      events.push_back(event); 
    }
    clWaitForEvents(events.size(), &events[0]);

      // Display inputs in rows
      std::cout << "Input Data" << std::endl;
      for (unsigned elems = 0; elems <  NUM_BUFFER_ELEMENTS; elems++){
          std::cout << " " << inputOutput[elems];
      }
      std::cout << std::endl;

    // Read back computed data
    clEnqueueReadBuffer(
        queue,
        out_buffer,
        CL_TRUE,
        0,
        sizeof(float) * NUM_BUFFER_ELEMENTS,
        (void*)inputOutput,
        0,
        NULL,
        NULL);
    // Display Outputs in rows
    std::cout << "Out Data" << std::endl;
    for (unsigned elems = 0; elems <  NUM_BUFFER_ELEMENTS; elems++)
    {
      std::cout << " " << inputOutput[elems];
    }
    std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;

    return 0;
}
