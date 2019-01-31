#include <Python.h>
#include "numpy/arrayobject.h"


static PyObject *SpamError;

static struct PyModuleDef spammodule;

static PyObject *
spam_system(PyObject *self, PyObject *args);

static PyObject *
example_wrapper(PyObject *dummy, PyObject *args);

static PyMethodDef SpamMethods[] = {

    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {"numpyex", example_wrapper, METH_VARARGS, 
     "abuse a numpy array"},

    {NULL, NULL, 0, NULL}        /* Sentinel */
};




static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);

    if (sts < 0) {
        PyErr_SetString(SpamError, "System command failed");
        return NULL;
    }

    return PyLong_FromLong(sts);
}
static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};
void mandelbrot(double * __restrict a_ptr, double * __restrict b_ptr, double * __restrict output_ptr, int dim_x, int dim_y){
    int i;
#pragma omp parallel for
    for(i = 0; i < dim_x; ++i){
        for(int j = 0; j < dim_y; j++){
            double count = 0;
            double a = 0;
            double b = 0;
            double temp_a;
            double aa = a_ptr[dim_y * i + j];
            double bb = b_ptr[dim_y * i + j];
            while (a * a + b * b < 4 && count < 2000) {
                temp_a = a * a - b * b + aa;
                b = 2 * a * b  + bb;
                a = temp_a;
                count++;
            }
            output_ptr[dim_y * i + j] = count;
        }
    }
}
static PyObject *
example_wrapper(PyObject *dummy, PyObject *args)
{
    PyObject *arg1=NULL, *arg2=NULL, *out=NULL;
    PyObject *arr1=NULL, *arr2=NULL, *oarr=NULL;

    if (!PyArg_ParseTuple(args, "OOO!", &arg1, &arg2,
        &PyArray_Type, &out)) return NULL;

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL) goto fail;
    oarr = PyArray_FROM_OTF(out, NPY_DOUBLE, NPY_INOUT_ARRAY);
    if (oarr == NULL) goto fail;

    if (
        (PyArray_NDIM(arr1) != 2) ||
        (PyArray_NDIM(arr2) != 2) ||
        (PyArray_NDIM(oarr) != 2)) {
        PyErr_SetString(SpamError, "All arrays must be 2D");
        goto fail;
    }

    /* code that makes use of arguments */
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */
    double * a_ptr = (double *) PyArray_DATA(arr1);
    double * b_ptr = (double *) PyArray_DATA(arr2);
    double * output_ptr = (double *) PyArray_DATA(oarr);
    int dim_x = PyArray_DIMS(arr1)[0];
    int dim_y = PyArray_DIMS(arr1)[1];

    mandelbrot(a_ptr, b_ptr, output_ptr, dim_x, dim_y);

    Py_DECREF(arr1);
    Py_DECREF(arr2);
    Py_DECREF(oarr);
    Py_INCREF(Py_None);
    return Py_None;

 fail:
    Py_XDECREF(arr1);
    Py_XDECREF(arr2);
    PyArray_XDECREF_ERR(oarr);
    return NULL;
}



PyMODINIT_FUNC
PyInit_spam(void)
{
    PyObject *m;

    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_INCREF(SpamError);
    PyModule_AddObject(m, "error", SpamError);
    import_array()
    return m;
}