package main

import (
	"fmt"
	"log"
	"time"

	ort "github.com/annotation-ai/onnxruntime_go"
)

func main() {
	// This line may be optional, by default the library will try to load
	// "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
	// You can download the library from:
	//     https://github.com/microsoft/onnxruntime/releases
	ort.SetSharedLibraryPath("libonnxruntime_1.14.1_osx_arm64.dylib")

	err := ort.InitializeEnvironment()
	if err != nil {
		log.Println(err)
	}
	defer ort.DestroyEnvironment()

	// To make it easier to work with the C API, this library requires the user
	// to create all input and output tensors prior to creating the session.
	inputData := []float32{
		0.6160029172897339,
		0.104542076587677,
		0.119082510471344,
		0.3446267247200012,
	}
	inputShape := ort.NewShape(1, 1, 4)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Println(err)
	}
	defer inputTensor.Destroy()

	// This hypothetical network maps a 2x5 input -> 2x3x4 output.
	outputShape := ort.NewShape(1, 1, 2)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Println(err)
	}
	defer outputTensor.Destroy()

	session, err := ort.NewSession[float32](
		"example_network.onnx",
		[]string{"input"},
		[]string{"output"},
	)
	if err != nil {
		log.Println(err)
	}
	defer session.Destroy()

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors. Simply
	// modify the input tensor's data (available via inputTensor.GetData())
	// before calling Run().
	start := time.Now()
	err = session.Run([]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		log.Println(err)
	}
	elapsed := time.Since(start)
	outputData := outputTensor.GetData()
	fmt.Println(outputData, elapsed)

	// Predict once again
	inputData = []float32{0.1, 0.2, 0.3, 0.4}
	inputTensor, err = ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Println(err)
	}
	defer inputTensor.Destroy()

	start = time.Now()
	err = session.Run([]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		log.Println(err)
	}
	elapsed = time.Since(start)
	outputData = outputTensor.GetData()
	fmt.Println(outputData, elapsed)

	// Predict once again
	inputData = []float32{0.4, 0.3, 0.2, 0.1}
	inputTensor, err = ort.NewTensor(inputShape, inputData)
	if err != nil {
		log.Println(err)
	}
	defer inputTensor.Destroy()

	start = time.Now()
	err = session.Run([]*ort.Tensor[float32]{inputTensor}, []*ort.Tensor[float32]{outputTensor})
	if err != nil {
		log.Println(err)
	}
	elapsed = time.Since(start)
	outputData = outputTensor.GetData()
	fmt.Println(outputData, elapsed)
}
