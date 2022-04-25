__kernel void averagebuffer(__global float * buffer,const int bufferWidth, const int bufferHeight,__global float* result)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
  
    for(int r=0;r<bufferHeight;r++){
                                     
        for(int c=0; c<bufferWidth;c++){
            
            result[0]+=buffer[r*bufferWidth+c]/(bufferWidth*bufferHeight);
        }
    }
}