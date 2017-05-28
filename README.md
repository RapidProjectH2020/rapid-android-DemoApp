# RAPID - Android Demo Application
This is part of the [RAPID Project](http://www.rapid-project.eu) and is an ongoing work.
```
RAPID enables automatic computation offloading of heavy tasks on Android and Linux/Windows Java applications.
Moreover, RAPID enables the possibility for embedding CUDA code in applications for generic Android devices
and for Java Linux/Windows.
```
While RAPID envisions to support heterogeneous devices, this is the demonstration of the tasks offloading on **Android**.
This demo uses the [RAPID Android Offloading Framework](https://github.com/RapidProjectH2020/rapid-android).  
The demo application shows three representative use case offloading scenarios:
* Android Java method offloading.
  * This is the simplest case of computation offloading, dealing with remote execution of Java methods.
  We have selected the [N-Queens puzzle](https://developers.google.com/optimization/puzzles/queens) as a representative for this use case.
  The N-Queens puzzle is the task of *arranging N chess queens in the chess keyboard so that no two queens can attack each other*.
  The current implementation is a brute force algorithm.
  The user can vary the number of queens from 4 to 8, changing the difficulty of the problem and its duration.
  When pressing the button "Solve Nqueens", the computation will be performed via the RAPID AC locally on the device or remotely on the VM.
  Cumulative statistics in terms of number of local/remote executions and average duration of local/remote executions will be shown to the user in real time.
  The expected result is that while increasing the number of queens, the gap between the local and remote execution should increase.

* Android C/C++ native function offloading.
  * Android allows developers to include native C/C++ code in their applications for increasing the performance 
  of intensive tasks or for allowing code reusability. 
  A normal Java method can call a native function thanks to the Java Native Interface (JNI). 
  To show that RAPID supports offloading of native functions, we have included in the demo a simple application 
  that simply returns the string "Hello from JNI" implemented in C++ and included as a native library in the demo application.
  Also in this case, the user can see cumulative statistics in terms of number and duration of local/remote execution.
  The expected result here is that the local execution will always be faster than the remote one, 
  given that the native function is not computationally intensive, meaning that the remote execution is penalized by the data transmission.
  However, this is just a simple demo that users can use as a starting point for building their applications 
  including offloadable native functions.

* Android CUDA programming and Android CUDA offloading.

In this page we will guide you on how to:
* [Quickly Install and Test the Demo Application](#installing-and-testing-the-demo).
* [Start Developing Android Applications with RAPID Offloading Support](#developing-android-applications-with-rapid-offloading-suppport).

For Java and CUDA code offloading on other platforms, have a look at the generic [RAPID Linux/Windows Demo Application](https://github.com/RapidProjectH2020/rapid-linux-DemoApp).

## Installing and Testing the Demo


## Developing Android Applications with RAPID Offloading Support
