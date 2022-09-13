# Load the module and generate the functions
#
using PyCall
const cv2 = pyimport("cv2")

module CppHello
  using CxxWrap
  @wrapmodule(joinpath(@__DIR__,"libhello"))

  function __init__()
    @initcxx
  end
end

mutable struct RetVal
  a::Int
  b::cv2::OpenVideo
end 
aa = Int
juliabb = RetVal

# Call greet and show the result
@show CppHello.greet()
@show CppHello.lib_fnctn()

