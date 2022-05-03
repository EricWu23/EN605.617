
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "testharness.h"
#include "supportOpencl.h"
#include "assignment.h"

const char kernel_names[5][20] = {
                         "vectorAdd",
                         "vectorSubtract",
                         "vectorMult",
                         "vectorDiv",
                         "vectorPow"
                     };

int testKernel(char** argv){

    std::cout << argv[0] << std::endl;
    std::cout << argv[1] << std::endl;
    std::cout << argv[2] << std::endl;
    std::cout << argv[3] << std::endl;

    cl_context context = 0;
    cl_command_queue commandQueue = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel0,kernel1,kernel2,kernel3 = 0;
    cl_mem memObjects[3] = { 0, 0, 0 };
    cl_int errNum;
    char * kernel_name;

  // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
     // Create a command-queue on the first device available
    // on the created context
    commandQueue = CreateCommandQueue(context, &device);
    if (commandQueue == NULL)
    {
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }

    // Create OpenCL program from MathKernelskernel source
    program = CreateProgram(context, device, "MathKernels.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }

    // Create OpenCL kernel
    kernel0 = clCreateKernel(program, "vectorAdd", NULL);
    if (kernel0 == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }
    kernel1 = clCreateKernel(program, "vectorSubtract", NULL);
    if (kernel1 == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel1, memObjects);
        return 1;
    }
    kernel2 = clCreateKernel(program, "vectorMult", NULL);
    if (kernel2 == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel2, memObjects);
        return 1;
    }
    kernel3 = clCreateKernel(program, "vectorDiv", NULL);
    if (kernel3 == NULL)
    {
        std::cerr << "Failed to create kernel" << std::endl;
        Cleanup(context, commandQueue, program, kernel3, memObjects);
        return 1;
    }
     // Create memory objects that will be used as arguments to
    // kernel.  First create host memory arrays that will be
    // used to store the arguments to the kernel
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }

    // Set the kernel arguments (result, a, b)
    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel0, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel0, 2, sizeof(cl_mem), &memObjects[2]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel arguments." << std::endl;
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }

    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };

    // Queue the kernel up for execution across the array
    cl_event event;
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel0, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    0, NULL, &event);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }
    clWaitForEvents(1, &event);
    clFinish(commandQueue);
    cl_ulong time_start;cl_ulong time_end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    double nanoSeconds = time_end-time_start;


    size_t kernel_name_size;

     clGetKernelInfo(
                    kernel0,
    CL_KERNEL_FUNCTION_NAME,
                          0,
                        NULL,
          &kernel_name_size);
          
    kernel_name=(char*)alloca(kernel_name_size);

    clGetKernelInfo(
                    kernel0,
    CL_KERNEL_FUNCTION_NAME,
            kernel_name_size,
        (void*) kernel_name,
                      NULL);


    std::cout << "Kernel execution "<<kernel_name <<" took "<<nanoSeconds/1000000.0<<" milli seconds."<<std::endl;
    // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 0, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        Cleanup(context, commandQueue, program, kernel0, memObjects);
        return 1;
    }

    // Output the result buffer
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    Cleanup(context, commandQueue, program, kernel0, memObjects);
    Cleanup(context, commandQueue, program, kernel1, memObjects);
    Cleanup(context, commandQueue, program, kernel2, memObjects);
    Cleanup(context, commandQueue, program, kernel3, memObjects);
    return 0;
}