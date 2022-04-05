//Based on the work of Andrew Krepps
#include <stdio.h>
#include <stdlib.h>
#include "assignment.h"
#include "nvgraph_SSSP.h"

int main(int argc, char** argv)
{
    execute_gpu_nvGraphSSSP_test(argc,argv);
}
