# RAPID - Android Demo Application
This is part of the [RAPID Project](http://www.rapid-project.eu) and is an ongoing work. While RAPID envisions to support heterogeneous devices, this is the demonstration of the tasks offloading on **Android**. This demo uses the [RAPID Android Offloading Framework](https://github.com/RapidProjectH2020/rapid-android).  
For Java and CUDA code offloading on other platforms, have a look at the generic [RAPID Linux/Windows Demo Application](https://github.com/RapidProjectH2020/rapid-linux-DemoApp).

In this page we will guide you on how to:
* [Quickly Install and Test the Demo Application](#installing-and-testing-the-demo).
* [Start Developing Android Applications with RAPID Offloading Support](#developing-android-applications-with-rapid-offloading-support).

## Intro
RAPID enables automatic computation offloading of heavy tasks on Android and Linux/Windows Java applications.
Moreover, RAPID enables the possibility for embedding CUDA code in applications for generic Android devices
and for Java Linux/Windows.  
RAPID enables highly CPU- or GPU-demanding applications to be offered through physical or virtual devices with lower capabilities or resources than the applications require, potentially backed by remote accelerators of the same class (D2D) or higher classes (hierarchical mode).  
RAPID supports its acceleration service through code offloading to more capable devices or devices with more resources, when this is deemed necessary or beneficial.
<p align="center">
<img src="http://rapid-project.eu/files/rapid-arch.png" width="480">
</p>

### Terminology
* **User Device (UD):** is the low-power device (phone, e.g.) that will be accelerated by code offloading. In our scenario it will be a phone running Android (version 4.1+ is recommended).
* **Acceleration Client (AC):** is an Android library that enables code offloading on the Android applications.
* **Application (AP):** is the Android application that will be accelerated by the framework. This application includes the AC as a library and uses the AC's API and the RAPID programming model.
* **VM:** is a Virtual Machine running on virtualized software, with the same operating system as the UD. In our scenario it will be an Android-x86 instance (version 4.0+ is recommended) running on VirtualBox.
* **Acceleration Server (AS):** is an Android application that runs on the VM and is responsible for executing the offloaded code by the client.

## Installing and Testing the Demo

### Description of the Demo App
The demo application shows three representative use case offloading scenarios:

* **Android Java method offloading.**

  This is the simplest case of computation offloading, dealing with remote execution of Java methods.
  We have selected the [N-Queens puzzle](https://developers.google.com/optimization/puzzles/queens) as a representative for this use case.
  The N-Queens puzzle is the task of *arranging N chess queens in the chess keyboard so that no two queens can attack each other*.
  The current implementation is a brute force algorithm.
  The user can vary the number of queens from 4 to 8, changing this way the difficulty of the problem and its duration.
  When pressing the button `Solve Nqueens`, the computation will be performed via the RAPID AC locally on the device or remotely on the VM.
  Cumulative statistics in terms of number of local/remote executions and average duration of local/remote executions will be shown to the user in real time.
  The expected result is that while increasing the number of queens, the gap between the local and remote execution should increase,
  with the remote executions being faster for bigger number of queens.

* **Android C/C++ native function offloading.**

  Android allows developers to include native C/C++ code in their applications for increasing the performance 
  of intensive tasks or for allowing code reusability. 
  A normal Java method can call a native function thanks to the Java Native Interface (JNI). 
  To show that RAPID supports offloading of native functions, we have included in the demo a simple application 
  that simply returns the string "*Hello from JNI*" implemented in C++ and included as a native library in the demo application.
  Also in this case, the user can see cumulative statistics in terms of number and duration of local/remote execution.
  The expected result here is that the local execution will always be faster than the remote one, 
  given that the native function is not computationally intensive, meaning that the remote execution is penalized by the data transmission.
  However, this is just a simple demo serving as a starting point for building applications that include offloadable native functions.

* **Android CUDA programming and Android CUDA offloading.**

  The third showcase is the most complex one, including CUDA code offloading.
  The demo application in this case is a matrix multiplication performed using CUDA.
  Notice that CUDA development is not possible for the majority of Android devices.
  As such, the developer:
  * Implements her CUDA code in a development machine as if it were for another supported operating system,
  e.g. Linux, and generates the Parallel Thread Execution (PTX) file using the NVIDIA CUDA Compiler (nvcc).
  * Then, the PTX file has to be embedded in the `assets/cuda-kernels` folder of the Android application, 
  where the RAPID framework will look for loading the file during runtime.
  
  When the execution of the method containing CUDA calls is performed locally, if the client device does not have a GPU, 
  RAPID will offload the CUDA calls from the client device to RAPID AS,
  which will take care of running them on the physical GPU of the machine where it is deployed (i.e. the RAPID cloud). 
  When the execution of the method containing the CUDA calls is performed remotely, because it is offloaded by the RAPID AC,
  the CUDA calls will be executed by RAPID on the remote GPU.

<p align="center">
<img src="http://rapid-project.eu/files/rapid-android-demo1.png" width="240">
</p>

### Installing
The demo shows how portion of the application's code can be run locally on the device or can be offloaded on a remote VM.
Installation steps:
1. Clone this project in Android Studio.
2. Install the demo application in an Android device (a phone with Android 4.1+ is recommended).
3. Install the Android-x86 VM we provide on a computer that is reachable by the phone device (i.e. the phone can ping the VM).
   * Install [VirtualBox](https://www.virtualbox.org/) on the computer.
   * Download the Android-x86-6.0 VirtualBox image of the VM from the RAPID website [here](http://rapid-project.eu/files/android-x86-6.0-r3.ova).
   (If you are having problems with this image then you can download the Android-x86-4.4 RAPID image from [here](http://rapid-project.eu/files/android-x86-4.4.ova))
   * [Import the image on VirtualBox](https://docs.oracle.com/cd/E26217_01/E26796/html/qs-import-vm.html).
   * Launch the Android VM you just imported.
     * The VM will automatically start the AS, you don't have to do anything.
     * Get the IP of the VM (one way to do this is to open the Terminal app inside the Android-x86 VM and type `busybox ifconfig`. If this command doesn't work then try `netcfg`.).  
     * Make sure that the phone device can ping the VM **and** the VM can ping the phone.
     * If you are experiencing issues with networking, you can read more about [VirtualBox 
     networking](https://www.virtualbox.org/manual/ch06.html).
   * ***Notice:** In the final release of the RAPID architecture we will provide VMs running on the RAPID cloud,
   meaning that you will not have to deal with these steps yourself.*
4. On the phone, select the radio button `Direct connection to VM` and write the **IP of the VM** on the text box that will open
(see the first figure below).
5. Press `Start` and wait until the app connects with the AS running on the VM.
   * A **green text** will notify that the connection with the VM was **successful**.
   * A **red text** will notify that the connection with the VM was **not successful**.
6. You will be presented with an Android activity showing the three demo apps.
7. You can select the execution location of the tasks using the radio buttons:
   * `Always Local` will instruct the framework to always execute the tasks locally on the device (phone).
   * `Always Remote` will instruct the framework to always execute the tasks remotely on the VM.
   * `Energy and Delay` will instruct the framework to make dynamic decisions and choose the execution location (local or remote) so that to minimize the energy and execution time of each task.
8. The second figure below shows the N-Queens puzzle being executed locally on the device.
9. The third figure shows the statistics of running the N-Queens puzzle in the device and remotely on the VM.
    * You can see that running it remotely is almost 10 times faster.

<p align="center">
<img src="http://rapid-project.eu/files/rapid-android-demo2.png" width="160">
<img src="http://rapid-project.eu/files/rapid-android-demo3.png" width="160">
<img src="http://rapid-project.eu/files/rapid-android-demo4.png" width="160">
</p>

## Developing Android Applications with RAPID Offloading Support
### Normal Android/Java Method Offloading
Let's start with a simple Android/Java class `Factorial.java` that implements a **heavy method** `long factorial(int n)`,
which takes in a number `n` and returns the factorial of number `n!`:
```java
public class Factorial {
    public long factorial(int n) { 
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}
```

To make this class RAPID offloadable, we simply need to perform the following steps:
* Include the [RAPID libraries](https://bintray.com/rapidprojecth2020/rapid) in the *build.gradle* file (use the latest version releases):
  ```gradle
  dependencies {
      compile 'eu.project.rapid:rapid-android-ac:0.0.9'
      compile 'eu.project.rapid:rapid-gvirtus4a:0.0.2'
  }
  ```
* Next, modify the source code in the following way:
  * Make the `Factorial` class *extend* `Remoteable`, which is a RAPID abstract class.
    * The `Remoteable` class implements the `Serializable` Java [interface](https://docs.oracle.com/javase/7/docs/api/java/io/Serializable.html), which means that the `Factorial` class should also be serializable.
    * Moreover, the abstract method `copyState()` should be implemented. Leave it empty for the moment.
  * Add a `@Remote` Java annotation to the `factorial` method to indicate that this method should be considered for offloading
  (we can also add `QoS` annotations for more fine-grained quality of service controls but let's keep this example simple and ignore it).
  * Declare an instance variable `transient DFE controller;` and initialize it on the constructor 
  with a `DFE` object that will be passed to the class from the main class.
    ```java
    import eu.project.rapid.ac.DFE;
    import eu.project.rapid.ac.Remoteable;

    public class Factorial extends Remoteable {

        private transient DFE controller;

        public Factorial(DFE controller) {
            this.controller = controller;
        }

        @Remote
        public long factorial(int n) { 
            long result = 1;
            for (int i = 2; i <= n; i++) {
                result *= i;
            }
            return result;
        }

        @Override
        public void copyState(Remoteable state) {
        }
    }
    ```
  * Next, download the RAPID source-to-source *compiler* to convert the modified class on RAPID-offloadable class.
    * Download the compiler as an executable jar file from the RAPID website [here](http://www.rapid-project.eu/files/rapid-acceleration-compiler.jar).
    * Run the compiler:
      ```bash
      java -jar rapid-acceleration-compiler.jar [<PATH_TO_THE_PROJECT> | <PATH_TO_THE_FILE>]
      ```
      * If the command line argument is a folder, all Java files inside the folder will be processed.
      * If the command line argument is a Java file, only that file will be processed.
      * The `Factorial.java` will be modified as shown below:
        ```java
        import eu.project.rapid.ac.DFE;
        import eu.project.rapid.ac.Remoteable;

        public class Factorial extends Remoteable {

            private transient DFE controller;

            public Factorial(DFE controller) {
                this.controller = controller;
            }

            @Remote
            public long localfactorial(int n) {
                long result = 1;
                for (int i = 2; i <= n; i++) {
                    result *= i;
                }
                return result;
            }

            @Override
            public void copyState(Remoteable state) {
            }

            public  long factorial (int n) {
               Method toExecute;
               Class<?>[] paramTypes = {int.class};
               Object[] paramValues = { n};
               long result = null;
               try {
                   toExecute = this.getClass().getDeclaredMethod("localfactorial", paramTypes);
                   result = (long) controller.execute(toExecute, paramValues, this);
               } catch (SecurityException e) {
                   // Should never get here
                   e.printStackTrace();
                   throw e;
               } catch (NoSuchMethodException e) {
                   // Should never get here
                   e.printStackTrace();
               } catch (Throwable e) {
                   // TODO Auto-generated catch block
                   e.printStackTrace();
               }
               return result;
           }
        }
        ```
* Create a `DFE dfe;` object in the main activity of your app (have a look at the `DemoActivity.java` 
[file](rapidDemoApp/src/main/java/eu/project/rapid/demo/DemoActivity.java) in this repo).
  * The `DFE` class is implemented as [Java Singleton](https://en.wikipedia.org/wiki/Singleton_pattern) and exposes
  two static public methods for getting an instance:
    * The first allows you to specify the IP of the VM where the AS is running.
    * The second performs a registration with the RAPID architecture and allocates a VM on the RAPID cloud
    (**currently works only for RAPID internal testing)**.
  * Pass the `DFE` object to the constructor of the `Factorial` when you create the new object.
  * Call the `dfe.onDestroy()` method in the `onDestroy()` method of your main activity.
* That's it, now when you run your application, the method `factorial()` will be executed via the `DFE` locally on the device
or remotely on the VM.
  
### Native C/C++ Android Code Offloading
### CUDA Android Support and CUDA Code Offloading
