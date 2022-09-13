/*
Author: Arkadiusz Grucelski

The file belong to AGICortex (to use and modify).

Main file for testing purposes (performance loss/other issues)
of CxxWrap in Julia.

\todo as in lib file, some more grained buld is needed at the moment
*/
#include <string>
#include <iostream>
//#include <opencv/cv.h>
//#include <opencv2/video/tracking.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>

#ifdef CxxJULIA
#include "jlcxx/jlcxx.hpp"
#endif 		 // CxxJULIA
#include"tst.h"

int main( int argc,char** argv )
{
  std::cout << greet();
  /*
   * beginning of tests 
   * */
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  float** src;  
  float** dst;  
  float** krnl; 
  src = new float*[WIDTH];
  dst = new float*[WIDTH];
  krnl = new float*[3];
  for (int i=0; i<WIDTH; i++){
     src[i] = new float[HEIGHT];
     dst[i] = new float[HEIGHT];
     if (i<KERNEL_SIZE) krnl[i] = new float[KERNEL_SIZE];
  }
  fillData( src,krnl );
  std::chrono::steady_clock::time_point initData = std::chrono::steady_clock::now();

  convolve( src,dst,krnl );
  std::chrono::steady_clock::time_point convolution = std::chrono::steady_clock::now();

  compareData( src,dst );
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  delete[] src;
  delete[] dst;
  delete[] krnl;
  /*
   * end of tests 
   * */
  std::cout << "(AG) Total time: " 
    << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
    << " (where init data " << std::chrono::duration_cast<std::chrono::milliseconds>(initData-begin).count() 
    << ", convolution " << std::chrono::duration_cast<std::chrono::milliseconds>(convolution-initData).count() 
    << ", conparison " << std::chrono::duration_cast<std::chrono::milliseconds>(end-convolution).count() 
    << ") [ms]" << std::endl;

  lib_fnctn();
  return 1;
}
