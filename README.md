Cross-Platform `onnxruntime` Wrapper for Go
===========================================

About
-----

This library seeks to provide an interface for loading and executing neural
networks from Go(lang) code, while remaining as simple to use as possible.

The [onnxruntime](https://github.com/microsoft/onnxruntime) library provides a
way to load and execute ONNX-format neural networks, though the library
primarily supports C and C++ APIs.  Several efforts exist to have written
Go(lang) wrappers for the `onnxruntime` library, but as far as I can tell, none
of these existing Go wrappers support Windows. This is due to the fact that
Microsoft's `onnxruntime` library assumes the user will be using the MSVC
compiler on Windows systems, while CGo on Windows requires using Mingw.

This wrapper works around the issues by manually loading the `onnxruntime`
shared library, removing any dependency on the `onnxruntime` source code beyond
the header files.  Naturally, this approach works equally well on non-Windows
systems.

Additionally, this library uses Go's recent addition of generics to support
multiple Tensor data types; see the `NewTensor` or `NewEmptyTensor` functions.

Requirements
------------

To use this library, you'll need a version of Go with cgo support.  If you are
not using an amd64 version of Windows or Linux (or if you want to provide your
own library for some other reason), you simply need to provide the correct path
to the shared library when initializing the wrapper.  This is seen in the first
few lines of the following example.


Example Usage
-------------

[The example](example/main.go) illustrates how this library can be used to load and run
an ONNX network taking a single input tensor and producing a single output
tensor, both of which contain 32-bit floating point values.

The execution results:
```bash
[1.184287 0.6071364] 48.917µs
```

The full documentation can be found at [pkg.go.dev](https://pkg.go.dev/github.com/yalue/onnxruntime_go).

