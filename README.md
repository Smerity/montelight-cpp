Montelight
==========

![](http://smerity.com/montelight-cpp/img/montage_passes_annotated.png)

*Faster raytracing through importance sampling, rejection sampling, and variance reduction*

For a broad overview of the approach employed, refer to our [website](http://smerity.com/montelight-cpp/).
For a guided exploration of the code, similar in style to an IPython Notebook, refer to our [annotated code overview](http://smerity.com/montelight-cpp/code_overview.html).

As this library doesn't use any external libraries, compilation is relatively simple.
To compile, use any C++11 compatible compiler such as `g++` or `clang++`:

    g++ -O3 -std=c++0x montelight.cc -o montelight
    clang++ -O3 -std=c++0x montelight.cc -o montelight

To run the program, simply execute it on the command line via:
    ./montelight

If you provide a `temp` folder, a number of in progress renders will be saved.
The final result will be saved as `render.ppm`.

For modifying the parameters of the program, refer to the annotated code overview.
