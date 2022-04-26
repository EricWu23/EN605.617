#pragma once

extern void checkErr(cl_int err, const char * name);

extern cl_int GetNumOfPlatforms(cl_uint* numPlatforms);

extern cl_int GetPlatformIDs(cl_uint numPlatforms,cl_platform_id * platformIDs);

extern cl_int GetNumOfAvaliableDevices(cl_uint* numDevices,cl_platform_id platform,cl_device_type device_type);

extern cl_int GetDeviceIDs(cl_platform_id platform,cl_device_type device_type,cl_uint num_entry,cl_device_id *devices,cl_uint *num_devices);

extern cl_context CreateContext(cl_platform_id PlatformId,cl_uint num_devices,const cl_device_id *devices);

extern cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);

extern cl_command_queue CreateCommandQueue(cl_context context, cl_device_id device);

