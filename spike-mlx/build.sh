#!/bin/sh
# Build the MLX spike harness. The Command Line Tools SDK 27 cannot link (unknown architecture arm64e.x1 in .tbd files); use Xcode.
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
SDK=$DEVELOPER_DIR/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.4.sdk
export SDKROOT=$SDK
set -e
xcrun clang++ -isysroot $SDK -O2 -std=c++17 -I. -I/opt/homebrew/include -c main.cpp tabfm_mlx_graph.cpp tabfm_onnx_reader.cpp
xcrun clang++ -isysroot $SDK -o mlxrun main.o tabfm_mlx_graph.o tabfm_onnx_reader.o -L/opt/homebrew/lib -lmlxc -lmlx -Wl,-rpath,/opt/homebrew/lib -framework Metal -framework Foundation -framework Accelerate
