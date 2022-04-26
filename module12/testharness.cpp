
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "info.hpp"
#include "testharness.h"
#include "supportOpencl.h"

#define DEFAULT_PLATFORM 0
#define DEFAULT_USE_MAP false

#define NUM_BUFFER_ELEMENTS 16
#define SUB_BUFFER_WIDTH 2
#define SUB_BUFFER_HEIGHT 2


void testHarness(){

    cl_int errNum;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;
    std::vector<cl_mem> buffers;//vector storing the subbuffers for input buffer
    std::vector<cl_mem> outputbuffers;//vector storing the subbuffers for output buffer
    cl_uint subbuffer_width=SUB_BUFFER_WIDTH;
    cl_uint subbuffer_height=SUB_BUFFER_HEIGHT;
    float * inputOutput;//host memmory that reused for the input and the output
    cl_uint numPlatforms;
    std::cout << "Assignment 12: 2x2 averagebuffer as moving average" << std::endl;
    
    // First, select an OpenCL platform to run on. 
    int platform = DEFAULT_PLATFORM; 
   
    errNum=GetNumOfPlatforms(&numPlatforms);
    platformIDs = (cl_platform_id *)alloca(sizeof(cl_platform_id) * numPlatforms);
    errNum=GetPlatformIDs(numPlatforms,platformIDs);
    DisplayPlatformInfo(platformIDs[platform], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
    
    // Second,Check avaliable Devices on the platform and select a device
    deviceIDs = NULL;
    errNum = GetNumOfAvaliableDevices(&numDevices,platformIDs[platform],CL_DEVICE_TYPE_ALL);
    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = GetDeviceIDs(platformIDs[platform],CL_DEVICE_TYPE_ALL,numDevices,&deviceIDs[0],NULL);
    
    // Third, Create a context associated with a platform and a bunch of Devices
    context = CreateContext(platformIDs[platform],numDevices,deviceIDs);
    
    // Fourth, Create a program object that will be built for a specific device within a specific context using a specific source file
    program = CreateProgram(context,deviceIDs[0],"mathkernel.cl");
    
    // Fifth, create a kernel object using the program object
    cl_kernel kernel = clCreateKernel(program,"averagebuffer",&errNum);
    checkErr(errNum, "clCreateKernel(averagebuffer)");
    
    // Sixth, Create command queues on a specific device in a specific context
    InfoDevice<cl_device_type>::display(deviceIDs[0], CL_DEVICE_TYPE, "CL_DEVICE_TYPE");
    cl_command_queue queue = CreateCommandQueue(context, deviceIDs[0]);

    //Seventh, Create memory objects
        // create a single buffer to cover all the input data, data will be filled later
        // for our purpose, we have to pad subbuffer_width*subbuffer_height-1 zeros
    uint elementsafterpadding=NUM_BUFFER_ELEMENTS+SUB_BUFFER_WIDTH*SUB_BUFFER_HEIGHT-1;
    
 
    inputOutput = new float[elementsafterpadding];//host
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    { 
        inputOutput[i] = (float)i+1;
    }
    //zero padding
    for(unsigned int i=NUM_BUFFER_ELEMENTS;i<elementsafterpadding;i++){
        inputOutput[i] = (float)0.0;
    }
    
   // create buffers and sub-buffers
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
        sizeof(float) * elementsafterpadding,
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
            CL_MEM_READ_WRITE,
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
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION,
            &region,
            &errNum);
        checkErr(errNum, "clCreateSubBuffer");

        outputbuffers.push_back(buffer);
    }
    
    std::vector<cl_event> events;
    size_t gWI =1 ;
     
    // Set up kernel arguments and call kernel for each subbuffer
    for (unsigned int i = 0; i < NUM_BUFFER_ELEMENTS; i++)
    {
      errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
      errNum |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &subbuffer_width);
      errNum |= clSetKernelArg(kernel, 2, sizeof(cl_uint), &subbuffer_height);
      errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputbuffers[i]);
      checkErr(errNum, "clSetKernelArg(averagebuffer)");

      cl_event event;
      errNum = clEnqueueNDRangeKernel(
          queue, 
          kernel, 
          1, 
          NULL,
          (const size_t*)&gWI, 
          NULL, 
          0, 
          NULL, 
          &event);
      events.push_back(event); 
    }
    
    clWaitForEvents(events.size(), &events[0]);
    
    cl_ulong time_start; cl_ulong time_end;
    clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(events[events.size()-1], CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;
    std::cout << "Kernel execution "<<"convolve" <<" took "<<nanoSeconds/1000000.0<<" milli seconds."<<std::endl;

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
    std::cout << "Output Data" << std::endl;
    for (unsigned elems = 0; elems <  NUM_BUFFER_ELEMENTS; elems++)
    {
      std::cout << " " << inputOutput[elems];
    }
    std::cout << std::endl;

    std::cout << "Program completed successfully" << std::endl;

}