set term pngcairo lw 2 
set y2tics
set key right center
set xlabel "* (120,120) [pt]"
set xtics ("4" 1, "6" 3, "8" 5, "10" 7, "12" 9, "14" 11, "16" 13)
set ylabel "t [ms]"

set output "c_lib.png"
plot \
	"ccc" u 5 i 0 w lp t "total time", \
	"" u 9 i 0 w lp axis x1y2 t "alloc and init data", \
	"" u 11 i 0 w lp axis x1y2 t "math operations"

set term pngcairo lw 2 
set output "cxx_wrap.png"
plot \
	"ccc" u 4 i 1 w lp t "total time", \
	"" u 8 i 1 w lp axis x1y2 t "alloc and init data", \
	"" u 10 i 1 w lp axis x1y2 t "math operations"

set term pngcairo lw 2 
set output "ccall.png"
plot \
	"ccc" u 5 i 2 w lp t "total time", \
	"" u 9 i 2 w lp axis x1y2 t "alloc and init data", \
	"" u 11 i 2 w lp axis x1y2 t "math operations"

set term pngcairo lw 2 
unset y2tics
set output "initdata.png"
plot \
	"ccc" u 9 i 0 w lp t "C++ lib", \
	"" u 8 i 1 w lp t "CxxWrap", \
	"" u 9 i 2 w lp t "multi ccall"

set term pngcairo lw 2 
unset y2tics
set output "heavymath.pbg"
plot \
	"ccc" u 11 i 0 w lp t "C++ lib", \
	"" u 10 i 1 w lp t "CxxWrap", \
	"" u 11 i 2 w lp t "multi ccall"
