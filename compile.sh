CXX="g++"
CXXFLAGS="-std=c++11 -O3"

$CXX $CXXFLAGS -o encrypt -I . encrypt.cpp

$CXX $CXXFLAGS decrypt.cpp -o decrypt
