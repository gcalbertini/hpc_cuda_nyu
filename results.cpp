#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdio.h>

int main()
{

    std::cout << "Some results..." << std::endl;
    // https://docs.python.org/3.0/extending/extending.html#calling-python-functions-from-c
    // https://www.codeproject.com/articles/820116/embedding-python-program-in-a-c-cplusplus-code
    // https://sites.northwestern.edu/yihanzhang/2019/08/22/how-to-invoke-python-function-from-c/
    void generateX(long batch_size, long numpredictors)
    {
        PyObject *pName, *pModule, *pDict, *pClass, *pInstance;

        // Initialize the Python interpreter
        Py_Initialize();

        // Build the name object
        pName = PyUnicode_FromString("Adder");
        // Load the module object
        pModule = PyImport_Import(pName);
        // pDict is a borrowed reference
        pDict = PyModule_GetDict(pModule);
        // Build the name of a callable class
        pClass = PyDict_GetItemString(pDict, "Adder");
        // Create an instance of the class
        if (PyCallable_Check(pClass))
        {
            pInstance = PyObject_CallObject(pClass, NULL);
        }
        else
        {
            pInstance = nullptr;
            std::cout << "Cannot instantiate the Python class" << std::endl;
        }
        int sum = 0;
        int x;

        for (size_t i = 0; i < 5; i++)
        {
            x = rand() % 100;
            sum += x;
            PyObject_CallMethod(pInstance, "add", "(i)", x);
        }
        PyObject *pValue = PyObject_CallMethod(pInstance, "add", "(i)", x);
        if (pValue)
            Py_DECREF(pValue);
        else
            PyErr_Print();
        std::cout << "the sum via C++ is " << sum << std::endl;

        std::getchar();
        Py_Finalize();
    }
}