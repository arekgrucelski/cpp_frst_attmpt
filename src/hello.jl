# Load the module and generate the functions
#
using BenchmarkTools

mutable struct myTime
  strt::UInt8
  stp::UInt8
end 

module CppAGIc
  using CxxWrap
#  @wrapmodule(joinpath(@__DIR__,"libtst"))
  @wrapmodule(joinpath("/home/agrucelski/Dokumenty/git_mw/cpp_frst_attmpt/","libtst"))

  function __init__()
    @initcxx
  end
end

mTm = myTime(10,10)

# Call greet and show the result
@show CppAGIc.greet()
@elapsed CppAGIc.start_all()
#@show CppHello.lib_fnctn()

#
#
#
c=[1 2 3 4.; 5 6 7 8; 9 10 11 12; 13 14 15 16.]
ct=c #transpose(c)

using Libdl
lib = Libdl.dlopen("./libtst.so")
sym = Libdl.dlsym(lib,:check_2darray)
display(@elapsed ccall(sym,Cvoid, (Ptr{Cdouble},Cint,Cint),c,4,4))
println(c)
#

fillData = Libdl.dlsym(lib,:fillData)
convolve = Libdl.dlsym(lib,:convolve)
compare = Libdl.dlsym(lib,:compareData)
tm_cmpDt = 0.0; tm_cnvDt = 0.0; tm_fllDt = 0.0
for ii in 1:50
  global tm_fllDt += @elapsed begin
    src=Array{Float32,2}(undef,1000,1000) #zeros(Float32,1000,1000)
    dst=Array{Float32,2}(undef,1000,1000) #zeros(Float32,1000,1000)
    krl=Array{Float32,2}(undef,3,3) #zeros(Float32,3,3)
    ccall(fillData,Cvoid, (Ptr{Cfloat},Ptr{Cfloat}),src,krl)
  end
  global tm_cnvDt += @elapsed ccall(convolve,Cvoid, (Ptr{Cfloat},Ptr{Cfloat},Ptr{Cfloat}),src,dst,krl)
  global tm_cmpDt += @elapsed begin
    ccall(compare,Cvoid, (Ptr{Cfloat},Ptr{Cfloat}),src,dst)
  end 
end 


println(1000.0*tm_fllDt/50)
println(1000.0*tm_cnvDt/50)
println(1000.0*tm_cmpDt/50)
#ccall((:check_2darray,"libtst.so"), Cvoid, (Ptr{Cfloat},Cint,Cint),ct,4,4)
#@show CppAGIc.check_2darray(ct,4,4)

