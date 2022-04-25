__kernel void averagebuffer(__global float * buffer,const int bufferWidth, const int bufferHeight,__global float* result)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    if(x<bufferWidth && y<bufferHeight){
      result[0] += buffer[y*bufferWidth+x]/(bufferWidth*bufferHeight);
    }
}