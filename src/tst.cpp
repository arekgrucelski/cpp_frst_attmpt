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
#ifdef CxxJULIA
#include "jlcxx/jlcxx.hpp"
#endif 		// CxxJULIA

std::string greet()
{
   return "hello, world\n";
}
extern int lib_fnctn( void )
{
  std::cout << "\nThis function seems to work" << 1 << "\n" ;
//    "OpenCV version " << CV_VERSION << std::endl;

  return 1;
}
/*
 * fill array with random data 
 */
void fillData( float** src,float** krnl)
{
  for (int i = 1; i < WIDTH-1; i++) {
    for (int j = 1; j < HEIGHT-1; j++) {
      src[i][j] = round( (rand()%1000 - 500) /1000.0 );
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
	std::cout << "An error at " << i << ","<< j << std::endl
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
      for (int ik = 0; ik < KERNEL_SIZE; ik++) {
    	for (int jk = 0; jk < KERNEL_SIZE; jk++) {
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

void tac(myTime* time)
{
//      time.end = std::chrono::steady_clock::now();
      
      std::cout << "Time "
//	 << std::chrono::duration_cast<std::chrono::milliseconds>(time->end - time->begin).count()
       	<< std::endl;
}


#ifdef CxxJULIA
JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
  mod.method("greet", &greet);
  mod.method("lib_fnctn", &lib_fnctn);
}
#endif 		// CxxJULIA

