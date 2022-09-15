/*
Author: Arkadiusz Grucelski

The file belong to AGICortex (to use and modify).

Naive implementation of convolution for testing purposes 
on Julia (CxxWrap) and standalone C++ to measure the performance lost

\todo more grained build for better CxxWrap calls from Julia
*/
#include <string>
#include <iostream>
#include <tgmath.h>
//#include <opencv/cv.h>
//#include <opencv2/video/tracking.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
#include <chrono>
#include"tst.h"
//
//#ifdef CxxJULIA
#include "jlcxx/jlcxx.hpp"
//#endif 		// CxxJULIA
//
#ifndef myTime_struct__
#define myTime_struct__
struct myTime
{
//  extern std::chrono::high_resolution_clock::time_point 
    uint8_t strt;
//  extern std::chrono::high_resolution_clock::time_point 
    uint8_t stp;
};
#endif // myTime_struct__



void start_all()
{
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
}

void fillData( float** src,float** krnl)
{
  for (int i = 1; i < WIDTH-1; i++) {
    for (int j = 1; j < HEIGHT-1; j++) {
      src[i][j] = round( (rand()%1000 - 500) /1000.0 );
      if ( (i<KERNEL_SIZE) && (j<KERNEL_SIZE) ) {
	      krnl[i][j] = round( (rand()%1000 - 500) /1000.0 );
      }
  } }

  krnl[0][0] = 0.0;
  krnl[0][1] = 0.0;
  krnl[0][2] = 0.0;
  krnl[1][0] = 0.0;
  krnl[1][1] = 1.0;
  krnl[1][2] = 0.0;
  krnl[2][0] = 0.0;
  krnl[2][1] = 0.0;
  krnl[2][2] = 0.0;
  return;
}
/*
 * compare data for two arrays 
 * */
void compareData( float** src, float** dst )
{
  //
  // image loops
  for (int i = 2; i < WIDTH-2; i++) {
    for (int j = 2; j < HEIGHT-2; j++) {
      if (src[i][j] != dst[i][j]) { 
	std::cout << "An error at " << i << ","<< j ;
	std::cout << "src = " << src[i][j] << ", dst = " << dst[i][j] << std::endl
	  << "STOP" << std::endl; return; 
      }
    }
  }
  return;
}
/*
 * Naive convolution implementation for c++ vs Julia(C++) tests 
 * */
void convolve( float** src,float** dst,float** krnl ) //( float **src,float **dst,float **krnl )
{
  // image loops
  for (int i = 1; i < WIDTH-1; i++) {
    for (int j = 1; j < HEIGHT-1; j++) {
	// kernel loops
      for (int ik = 0; ik < KERNEL_SIZE-1; ik++) {
    	for (int jk = 0; jk < KERNEL_SIZE-1; jk++) {
	    // the body of convolution
	  dst[i][j] = dst[i][j] + src[i+ik-1][j+jk-1]*krnl[ik][jk];
      } } // ik jk loops
  } } // i j loops

/*  
  std::cout << krnl[0][0] << "," << krnl[1][0] << "," << krnl[2][0] << std::endl;
  std::cout << krnl[0][1] << "," << krnl[1][1] << "," << krnl[2][1] << std::endl;
  std::cout << krnl[0][2] << "," << krnl[1][2] << "," << krnl[2][2] << std::endl;
  */
  return;
}


/*
    Blck AG mdfctn of MZ code

*/
void tic( myTime* time) //void )
{
     return; // &std::chrono::steady_clock::now();
}

void tac( float* bb ) //myTime* time )
{
//      time.end = std::chrono::steady_clock::now();
      
      std::cout << "Time: " << *bb  << " , " << bb
//	 << std::chrono::duration_cast<std::chrono::milliseconds>(time->end - time->begin).count()
       	<< std::endl;
}


#ifdef CxxJULIA
JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.method("greet", &greet);
  mod.method("lib_fnctn", &lib_fnctn);
//  mod.map_type<myTime>("myTime");
  mod.method("tac",&tac);
  mod.method("start_all",&start_all);
}
#endif 		// CxxJULIA

std::string greet()
{
   return "hello, world\n";
}
extern int lib_fnctn( int i )
{
  std::cout << "\nThis function seems to work: " << i << "\n" ;
//    "OpenCV version " << CV_VERSION << std::endl;

  return 1;
}

