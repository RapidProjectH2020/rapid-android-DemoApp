# RAPID - Android Demo Application
This is part of the [RAPID Project](http://www.rapid-project.eu) and is an ongoing work. This demo uses the [RAPID Android Offloading Framework](https://github.com/RapidProjectH2020/rapid-android).
```
RAPID enables automatic computation offloading of heavy tasks on Android and Linux/Windows Java applications.
Moreover, RAPID enables the possibility for embedding CUDA code in applications for generic Android devices
and for Java Linux/Windows.
```
While RAPID envisions to support heterogeneous devices, this is the demonstration of the tasks offloading on **Android**.
For Java and CUDA code offloading on other platforms, have a look at the generic [RAPID Linux/Windows Demo Application](https://github.com/RapidProjectH2020/rapid-linux-DemoApp).

The demo application shows three representative use case offloading scenarios:
* **Android Java method offloading.**

  This is the simplest case of computation offloading, dealing with remote execution of Java methods.
  We have selected the [N-Queens puzzle](https://developers.google.com/optimization/puzzles/queens) as a representative for this use case.
  The N-Queens puzzle is the task of *arranging N chess queens in the chess keyboard so that no two queens can attack each other*.
  The current implementation is a brute force algorithm.
  The user can vary the number of queens from 4 to 8, changing the difficulty of the problem and its duration.
  When pressing the button "Solve Nqueens", the computation will be performed via the RAPID AC locally on the device or remotely on the VM.
  Cumulative statistics in terms of number of local/remote executions and average duration of local/remote executions will be shown to the user in real time.
  The expected result is that while increasing the number of queens, the gap between the local and remote execution should increase.

* **Android C/C++ native function offloading.**

  Android allows developers to include native C/C++ code in their applications for increasing the performance 
  of intensive tasks or for allowing code reusability. 
  A normal Java method can call a native function thanks to the Java Native Interface (JNI). 
  To show that RAPID supports offloading of native functions, we have included in the demo a simple application 
  that simply returns the string "Hello from JNI" implemented in C++ and included as a native library in the demo application.
  Also in this case, the user can see cumulative statistics in terms of number and duration of local/remote execution.
  The expected result here is that the local execution will always be faster than the remote one, 
  given that the native function is not computationally intensive, meaning that the remote execution is penalized by the data transmission.
  However, this is just a simple demo that users can use as a starting point for building their applications 
  including offloadable native functions.

* **Android CUDA programming and Android CUDA offloading.**

  The third showcase is the most complex one, including CUDA code offloading.
  The demo application in this case is a matrix multiplication performed using CUDA.
  Notice that CUDA development is not possible for the majority of Android devices.
  As such, the developer implements her CUDA code in a development machine as if it were for another supported operating system,
  e.g. Linux, and generates the Parallel Thread Execution (PTX) file using the NVIDIA CUDA Compiler (nvcc).
  Then, the PTX file has to be embedded in the assets/cuda-kernels folder of the Android application, 
  where the GVirtuS Frontend will look for loading the file during runtime.
  When the execution of the method containing CUDA calls is performed locally, if the client device does not have a GPU, 
  the GVirtuS Frontend will offload the CUDA calls from the client device to the GVirtuS Backend, 
  which will take care of running them on the physical GPU of the machine where it is deployed (i.e. the RAPID cloud). 
  When the execution of the method containing the CUDA calls is performed remotely, because it is offloaded by the RAPID AC,
  the CUDA calls will be executed by the GVirtuS Frontend on the remote VM.
  Given that the VMs do not have a physical GPU, also in this case the GVirtuS Frontend will offload the execution to the GVirtuS Backend.
  However, in this case the GVirtuS Frontend and Backend could potentially be running on the same physical machine 
  (when using the RAPID architecture), which means that the latency of each CUDA call will be smaller than the local case.
  Also in this case, the user can see the cumulative statistics as for the other applications.


In this page we will guide you on how to:
* [Quickly Install and Test the Demo Application](#installing-and-testing-the-demo).
* [Start Developing Android Applications with RAPID Offloading Support](#developing-android-applications-with-rapid-offloading-suppport).


## Installing and Testing the Demo


## Developing Android Applications with RAPID Offloading Support
