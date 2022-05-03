//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// HelloWorld.cpp
//
//    This is a simple example that demonstrates basic OpenCL setup and
//    use.


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "assignment.h"
#include "testharness.h"
///
//  Constants
//
const int ARRAY_SIZE = 1000;


///
//	main() for HelloWorld example
//
int main(int argc, char** argv)
{
   testKernel(*(kernel_names+0));
   testKernel(*(kernel_names+1));
   testKernel(*(kernel_names+2));
   testKernel(*(kernel_names+3));
   testKernel(*(kernel_names+4));
   return 0;
}
