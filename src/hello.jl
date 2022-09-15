# Load the module and generate the functions
#

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
#@show CppHello.lib_fnctn()

