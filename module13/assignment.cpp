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
#include <iostream>
///
//  Constants
//
const int ARRAY_SIZE = 10;


///
//	main() for HelloWorld example
//

int main(int argc, char** argv)
{
  int status;
   if (argc != 5){
        std::cerr << "USAGE: " << argv[0] << " 0 1 2 3" << std::endl;
        std::cout << "The 1 2 3 4 can be replaced by arbitrary permutation. For example: 2 3 1 0" << std::endl;
        std::cout << "0 stands for a=a+b" << std::endl;
        std::cout << "1 stands for a=a-b" << std::endl;
        std::cout << "2 stands for a=a*b" << std::endl;
        std::cout << "3 stands for a=a/b" << std::endl;
        return 1;
   }
   

   status=testKernel(argv);
   return status;
}
