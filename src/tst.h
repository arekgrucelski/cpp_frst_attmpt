#ifdef CxxJULIA
#include "jlcxx/jlcxx.hpp"
#endif 		 // CxxJULIA
#include<chrono>


#ifndef tst_h__
#define tst_h__

#define WIDTH 640
#define HEIGHT 480
#define KERNEL_SIZE 3

/*
 * \todo create time elapsed function 
 */
struct myTime
{
//  extern std::chrono::high_resolution_clock::time_point 
    long begin;
//  extern std::chrono::high_resolution_clock::time_point 
    long end;
};

extern void fillData( float**,float** );
extern void compareData( float**, float** );
extern void convolve( float** ,float** ,float** );
//
extern std::string greet();
extern int lib_fnctn( void );
extern void tic( myTime *);
extern void tac( myTime *);

#endif 		// tst_h__
