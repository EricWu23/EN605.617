#pragma once

extern void checkErr(cl_int err, const char * name);

extern cl_int GetPlatformIDs(cl_uint numPlatforms,cl_platform_id * platformIDs);