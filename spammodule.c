#include <Python.h>
#include "numpy/arrayobject.h"
#include "mandel.h"


static PyObject *SpamError;

static struct PyModuleDef spammodule;

static PyObject *
spam_system(PyObject *self, PyObject *args);

static PyObject *
example_wrapper(PyObject *dummy, PyObject *args);

static PyObject *
perturb_mandel(PyObject *dummy, PyObject *args);

static PyMethodDef SpamMethods[] = {

    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {"numpyex", example_wrapper, METH_VARARGS, 
     "abuse a numpy array"},
    {"perturb_mandel", perturb_mandel, METH_VARARGS, 
     "reference dc_real dc_imag out iterations"},
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
void mandelbrot(double * __restrict a_ptr, double * __restrict b_ptr, double * __restrict output_ptr, int dim_x, int dim_y, int iter){
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
            while (a * a + b * b < 4 && count < iter) {
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

    int iter = 5;

    if (!PyArg_ParseTuple(args, "OOO!i", &arg1, &arg2,
        &PyArray_Type, &out, &iter)) return NULL;

    printf("%d", iter);

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

    mandelbrot(a_ptr, b_ptr, output_ptr, dim_x, dim_y, iter);

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

void perturb_mandelbrot_func(double * __restrict z_ptr, double * __restrict dc_real_ptr, 
    double * __restrict dc_imag_ptr, double * __restrict output_ptr, int dim_x, int dim_y, int iter){
    int i;
#pragma omp parallel for
    for(i = 0; i < dim_x; ++i){
        for(int j = 0; j < dim_y; j++){
            int count = 0;
            double d_real = 0;
            double d_imag = 0;
            double temp_d_real;
            double dc_real = dc_real_ptr[dim_y * i + j];
            double dc_imag = dc_imag_ptr[dim_y * i + j];

            while ((d_real + z_ptr[2 * count]) * (d_real + z_ptr[2 * count]) + 
                   (d_imag + z_ptr[2 * count + 1]) * (d_imag + z_ptr[2 * count + 1]) < 4 && 
                   count < iter) {
                double z_real = z_ptr[2 * count];
                double z_imag = z_ptr[2 * count + 1];
                temp_d_real = 2 * z_real * d_real - 2 * z_imag * d_imag + d_real * d_real - d_imag * d_imag + dc_real;
                d_imag = 2 * z_real * d_imag + 2 * z_imag * d_real + 2 * d_real * d_imag + dc_imag;
                d_real = temp_d_real;
                count++;
            }
            output_ptr[dim_y * i + j] = (double) count;
        }
    }
}

static PyObject *
perturb_mandel(PyObject *dummy, PyObject *args)
{
    PyObject *ref_r_arg=NULL, *ref_i_arg=NULL, *arg1=NULL, *arg2=NULL, *out=NULL;

    PyObject *ref_r_arr=NULL, *ref_i_arr=NULL, *arr1=NULL, *arr2=NULL, *oarr=NULL;

    int iter = 0;

    if (!PyArg_ParseTuple(args, "OOOOO!i", &ref_r_arg, &ref_i_arg, &arg1, &arg2,
        &PyArray_Type, &out, &iter)) return NULL;
    printf("%d\n", iter);

    arr1 = PyArray_FROM_OTF(arg1, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr1 == NULL) return NULL;
    arr2 = PyArray_FROM_OTF(arg2, NPY_DOUBLE, NPY_IN_ARRAY);
    if (arr2 == NULL) goto fail;
    oarr = PyArray_FROM_OTF(out, NPY_INT, NPY_INOUT_ARRAY);
    if (oarr == NULL) goto fail;
    ref_r_arr = PyArray_FROM_OTF(ref_r_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (ref_r_arr == NULL) goto fail;
    ref_i_arr = PyArray_FROM_OTF(ref_i_arg, NPY_DOUBLE, NPY_IN_ARRAY);
    if (ref_i_arr == NULL) goto fail;
    if (
        (PyArray_NDIM(arr1) != 2) ||
        (PyArray_NDIM(arr2) != 2) ||
        (PyArray_NDIM(oarr) != 2)) {
        PyErr_SetString(SpamError, "All arrays must be 2D");
        goto fail;
    }
    
    /* You will probably need at least
       nd = PyArray_NDIM(<..>)    -- number of dimensions
       dims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
                                     showing length in each dim.
       dptr = (double *)PyArray_DATA(<..>) -- pointer to data.

       If an error occurs goto fail.
     */

    double * ref_r_ptr = (double *) PyArray_DATA(ref_r_arr);
    double * ref_i_ptr = (double *) PyArray_DATA(ref_i_arr);
    double * a_ptr = (double *) PyArray_DATA(arr1);
    double * b_ptr = (double *) PyArray_DATA(arr2);
    int * output_ptr = (int *) PyArray_DATA(oarr);
    int dim_x = PyArray_DIMS(arr1)[0];
    int dim_y = PyArray_DIMS(arr1)[1];

    int copyback = (int)PyArray_DIMS(ref_r_arr)[0];
    printf("num2copy:%d", copyback);
    cu_mandel(ref_r_ptr, ref_i_ptr, a_ptr, b_ptr, iter, output_ptr, copyback);

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




#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define PREC 4

#define DBL_INT uint64_t
#define ONE_INT uint32_t

#define SHIFTBY 32

typedef struct FixNum {
    int64_t sign;
    ONE_INT dpart[PREC];
} FixNum;

void uadd(FixNum * a, FixNum * b, FixNum* out) {
    DBL_INT carry = 0;
    for(int j = PREC - 1; j >= 0; j-- ){
        DBL_INT res = ((DBL_INT) a->dpart[j]) + ((DBL_INT)b->dpart[j]) + ((DBL_INT) carry);
        out->dpart[j] = (ONE_INT) res;
        carry = res >> SHIFTBY;
    }
}

void usub(FixNum * a, FixNum * b, FixNum* out) {
    DBL_INT carry = 0;
    for(int j = PREC - 1; j >= 0; j-- ){
        DBL_INT t = a->dpart[j] - carry;
        if(t < b->dpart[j]){
          carry = 1;
          t += (DBL_INT)1 << SHIFTBY; 
        } else {
          carry = 0;
        }
        DBL_INT res = t - ((DBL_INT)b->dpart[j]);
        out->dpart[j] = (ONE_INT) res;
        
    }
}

int cmp(FixNum * a, FixNum * b ) {
    if(a->sign != b->sign) {
        return 2 * (a->sign > b->sign) - 1;
    }
    for(int i = 0; i < PREC; i++){
        if (a->dpart[i] != b->dpart[i] ){
            return 2 * (a->dpart[i] > b->dpart[i]) - 1;
        }
    }
    return 0;
}

int abscmp(FixNum * a, FixNum * b ) {
    for(int i = 0; i < PREC; i++){
        if (a->dpart[i] != b->dpart[i] ){
            return 2 * (a->dpart[i] > b->dpart[i]) - 1;
        }
    }
    return 0;
}

void add(FixNum * a, FixNum * b, FixNum* out) {
    if(a->sign == b->sign){
        uadd(a, b, out);
        out->sign = a->sign;
    } else {
        
        int t = abscmp(a, b);
        if (t == 1) {
            usub(a, b, out);
            out->sign = a->sign;
        } else {
            usub(b, a, out);
            out->sign = b->sign;
        }
    }

}


void mul(FixNum * a, FixNum* b, FixNum* out) {
    ONE_INT mul_temp[PREC * PREC * 2];
    int j;
    for(j = 0; j < PREC * PREC; j++ ){
        ((DBL_INT*) mul_temp)[j] = 0;
    }
    int i;
    //fill multiply table
    DBL_INT carry;
    for(i = 0; i < PREC; i++){
        carry = 0;
        for(j =  PREC - 1; j >= 0; j--){
            
            DBL_INT res = ((DBL_INT) a->dpart[j]) * ((DBL_INT) b->dpart[i]) + carry;
            mul_temp[i * 2 * PREC + i + j] = (ONE_INT) res;
            //printf("res: %02d ", res);
            carry = res >> SHIFTBY;
            //printf("carry: %02d ", carry);
        }
        mul_temp[i * 2 * PREC + i - 1] = carry;
        //printf("\n");
    }
    //sum
    carry = 0;

    for(int col = 2 * PREC - 1; col >= 0; col--){
        for(int row = 0; row < PREC; row++){
            carry = carry + mul_temp[row * 2 * PREC + col];
        }
        if(col < PREC){
            out->dpart[col] = (ONE_INT) carry;
        }
        carry = carry >> SHIFTBY;
    }

    out->sign = a->sign * b->sign;


}


FixNum* test() {
    FixNum a = {0, {2, 0, 2}};
    FixNum b = {0, {0, 1 << 31, 9}};

    FixNum* out;
    out = malloc(sizeof(out));

    usub(&a, &b, out);
    return out;

}

void unity(FixNum * a, FixNum * out){
    FixNum b = {{1, 0, 0}, 0};
    mul(a, &b, out);
}
int main(void) {
  printf("\n");
  FixNum z = *test();
  printf("--\n");
  for(int i = 0; i < PREC; i++){
    printf("%08X ", z.dpart[i]);
  }
  return 0;
}

