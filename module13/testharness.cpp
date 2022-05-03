
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
#include <string.h>

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

int testKernel(char** argv){
      
 	cl_int err=0;
    cl_context context = 0;
    cl_command_queue commandQueue0,commandQueue1,commandQueue2,commandQueue3 = 0;
    cl_program program = 0;
    cl_device_id device = 0;
    cl_kernel kernel0,kernel1,kernel2,kernel3 = 0;
    cl_mem memObjects[3] = { 0, 0};
    cl_int errNum;

  // Create an OpenCL context on first available platform
    context = CreateContext();
    if (context == NULL)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        return 1;
    }
    // Create five user defined events on the context
    cl_event uevent0=clCreateUserEvent(context,&err);
    checkErr(err, "clCreateUserEvent0");
    cl_event uevent1=clCreateUserEvent(context,&err);
    checkErr(err, "clCreateUserEvent1");
    cl_event uevent2=clCreateUserEvent(context,&err);
    checkErr(err, "clCreateUserEvent2");
    cl_event uevent3=clCreateUserEvent(context,&err); 
    checkErr(err, "clCreateUserEvent3");
       
     // Create a command-queue on the first device available on the created context
    commandQueue0 = CreateCommandQueue(context, &device,0);
    if (commandQueue0 == NULL)
    {
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }
    // Create a command-queue on the first device available on the created context
    commandQueue1 = CreateCommandQueue(context, &device,0);
    if (commandQueue1 == NULL)
    {
        Cleanup(context, commandQueue1, program, kernel0, memObjects);
        return 1;
    }
    
    // Create a command-queue on the first device available on the created context
    commandQueue2 = CreateCommandQueue(context, &device,0);
    if (commandQueue2 == NULL)
    {
        Cleanup(context, commandQueue2, program, kernel0, memObjects);
        return 1;
    }
    // Create a command-queue on the first device available on the created context
    commandQueue3 = CreateCommandQueue(context, &device,0);
    if (commandQueue3 == NULL)
    {
        Cleanup(context, commandQueue3, program, kernel0, memObjects);
        return 1;
    }

    // Create OpenCL program from MathKernelskernel source
    program = CreateProgram(context, device, "MathKernels.cl");
    if (program == NULL)
    {
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }

    // Create OpenCL kernel Object
    kernel0 = clCreateKernel(program, "vectorAdd", NULL);
    if (kernel0 == NULL)
    {
        std::cerr << "Failed to create kernel 0" << std::endl;
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }
    kernel1 = clCreateKernel(program, "vectorSubtract", NULL);
    if (kernel1 == NULL)
    {
        std::cerr << "Failed to create kernel 1" << std::endl;
        Cleanup(context, commandQueue1, program, kernel1, memObjects);
        return 1;
    }
    kernel2 = clCreateKernel(program, "vectorMult", NULL);
    if (kernel2 == NULL)
    {
        std::cerr << "Failed to create kernel 2" << std::endl;
        Cleanup(context, commandQueue2, program, kernel2, memObjects);
        return 1;
    }
    kernel3 = clCreateKernel(program, "vectorDiv", NULL);
    if (kernel3 == NULL)
    {
        std::cerr << "Failed to create kernel 3" << std::endl;
        Cleanup(context, commandQueue3, program, kernel3, memObjects);
        return 1;
    }
    
    
     // Create memory objects that will be used as arguments to kernels
    float result[ARRAY_SIZE];
    float a[ARRAY_SIZE];
    float b[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        a[i] = (float)i;
        b[i] = (float)(2);
    }

    if (!CreateMemObjects(context, memObjects, a, b))
    {
        std::cerr << "Failed to Create memoryObjects" << std::endl;
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }

    // Set the kernel arguments (a, b, result)
    errNum = clSetKernelArg(kernel0, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel0, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel0, 2, sizeof(cl_mem), &memObjects[0]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel0 arguments." << std::endl;
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }

    errNum = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel1, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel1, 2, sizeof(cl_mem), &memObjects[0]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel1 arguments." << std::endl;
        Cleanup(context, commandQueue1, program, kernel0, memObjects);
        return 1;
    }
    
    
    errNum = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel2, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel2, 2, sizeof(cl_mem), &memObjects[0]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel2 arguments." << std::endl;
        Cleanup(context, commandQueue2, program, kernel0, memObjects);
        return 1;
    }
    
    errNum = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &memObjects[0]);
    errNum |= clSetKernelArg(kernel3, 1, sizeof(cl_mem), &memObjects[1]);
    errNum |= clSetKernelArg(kernel3, 2, sizeof(cl_mem), &memObjects[0]);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error setting kernel3 arguments." << std::endl;
        Cleanup(context, commandQueue3, program, kernel0, memObjects);
        return 1;
    }    
    

    // Queue kernels up for execution on Device
    size_t globalWorkSize[1] = { ARRAY_SIZE };
    size_t localWorkSize[1] = { 1 };
    
        //enqueue the kernel 1
    cl_event event1;
    errNum = clEnqueueNDRangeKernel(commandQueue1, kernel1, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &uevent1, &event1);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue1, program, kernel0, memObjects);
        return 1;
    }
    
     //enqueue the kernel 2
    cl_event event2;
    errNum = clEnqueueNDRangeKernel(commandQueue2, kernel2, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &uevent2, &event2);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue2, program, kernel0, memObjects);
        return 1;
    }
 
   
    //enqueue the kernel 3
    cl_event event3;
    errNum = clEnqueueNDRangeKernel(commandQueue3, kernel3, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &uevent3, &event3);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue3, program, kernel0, memObjects);
        return 1;
    } 
    
    //enqueue the kernel 0
    cl_event event0;
    errNum = clEnqueueNDRangeKernel(commandQueue0, kernel0, 1, NULL,
                                    globalWorkSize, localWorkSize,
                                    1, &uevent0, &event0);
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error queuing kernel for execution." << std::endl;
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }
 
 // Events Processing Hub  
       for(int i=1; i<5;i++){
            if(!strcmp(argv[i], "0")){
                
                err=clSetUserEventStatus(uevent0,CL_COMPLETE);
                std::cout << "Adding Begin" << std::endl;
                err=clWaitForEvents(1,&event0);
                 std::cout << "Adding Finshed" << std::endl;

            }
            else if(!strcmp(argv[i], "1")){
                 err=clSetUserEventStatus(uevent1,CL_COMPLETE);
                 std::cout << "Subtracting begin" << std::endl;
                 err=clWaitForEvents(1,&event1);
                std::cout << "Subtracting Finished" << std::endl;
 
            }
            else if(!strcmp(argv[i], "2")){

                 err=clSetUserEventStatus(uevent2,CL_COMPLETE);
                 std::cout << "Multiplying begin" << std::endl;
                 err=clWaitForEvents(1,&event2);
                 std::cout << "Multiplying finished" << std::endl;

            }
            else if (!strcmp(argv[i], "3")){

                 err=clSetUserEventStatus(uevent3,CL_COMPLETE);
                 std::cout << "Dividing begin" << std::endl;
                 err=clWaitForEvents(1,&event3);
                 std::cout << "Dividing finished" << std::endl;
            }
            else{
                 std::cout << "invalid commandline argument. You should only input number among 0,1,2,3" << std::endl;
                 return 1;
            }
    }

 
    cl_event event_wait_list[4]={event0,event1,event2,event3};
        // Read the output buffer back to the Host
    errNum = clEnqueueReadBuffer(commandQueue0, memObjects[0], CL_TRUE,
                                 0, ARRAY_SIZE * sizeof(float), result,
                                 4, event_wait_list, NULL);
        if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error reading result buffer." << std::endl;
        std::cerr << "errNum:" << errNum<<std::endl;
        Cleanup(context, commandQueue0, program, kernel0, memObjects);
        return 1;
    }
     
    
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "Executed program succesfully." << std::endl;
    
    Cleanup(context, commandQueue0, program, kernel0, memObjects);
    return 0;
}