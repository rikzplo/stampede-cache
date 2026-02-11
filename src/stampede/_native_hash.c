/*
 * stampede/_native_hash.c
 *
 * C-accelerated structural FNV-1a hashing for stampede.
 *
 * Produces IDENTICAL output to the pure-Python implementation in hashing.py.
 * Hashes Python object trees directly without serialization — dicts and sets
 * use commutative XOR so insertion order never matters.
 *
 * Typically 10-50x faster than the pure-Python fallback for nested structures.
 *
 * Exposed functions:
 *   fast_hash(obj, length=16) -> str    # structural FNV-1a -> hex string
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ========================================================================= */
/* FNV-1a constants (must match Python implementation exactly)               */
/* ========================================================================= */

#define FNV_OFFSET  UINT64_C(0xCBF29CE484222325)
#define FNV_PRIME   UINT64_C(0x100000001B3)
#define GOLDEN      UINT64_C(0x9E3779B97F4A7C15)

/* Type tags — must match Python _sh() tag bytes */
#define TAG_NONE    0x00
#define TAG_TRUE    0x01
#define TAG_FALSE   0x02
#define TAG_INT     0x03
#define TAG_FLOAT   0x04
#define TAG_STR     0x05
#define TAG_BYTES   0x06
#define TAG_LIST    0x07
#define TAG_TUPLE   0x08
#define TAG_DICT    0x09
#define TAG_SET     0x0A

#define MAX_DEPTH   100

/* ========================================================================= */
/* Core FNV-1a primitives                                                    */
/* ========================================================================= */

static inline uint64_t fnv_byte(uint64_t h, uint8_t b)
{
    return (h ^ b) * FNV_PRIME;
}

static inline uint64_t fnv_bytes(uint64_t h, const uint8_t *data, Py_ssize_t len)
{
    for (Py_ssize_t i = 0; i < len; i++)
        h = (h ^ data[i]) * FNV_PRIME;
    return h;
}

static inline uint64_t fnv_u64(uint64_t h, uint64_t v)
{
    for (int i = 0; i < 8; i++) {
        h = (h ^ (v & 0xFF)) * FNV_PRIME;
        v >>= 8;
    }
    return h;
}

/* ========================================================================= */
/* Structural hash — recursive, mirrors Python _sh() exactly                 */
/* ========================================================================= */

static uint64_t sh(PyObject *val, int depth)
{
    if (depth > MAX_DEPTH) return 0;

    uint64_t h = FNV_OFFSET;

    /* None */
    if (val == Py_None)
        return fnv_byte(h, TAG_NONE);

    /* Bool (must check before int — bool is subclass of int in Python) */
    if (PyBool_Check(val))
        return fnv_byte(h, val == Py_True ? TAG_TRUE : TAG_FALSE);

    /* Int */
    if (PyLong_Check(val)) {
        h = fnv_byte(h, TAG_INT);
        int overflow;
        long long v = PyLong_AsLongLongAndOverflow(val, &overflow);
        if (!overflow && !PyErr_Occurred()) {
            return fnv_u64(h, (uint64_t)v);
        }
        PyErr_Clear();
        /* Big int: convert to string, hash the UTF-8 bytes */
        PyObject *s = PyObject_Str(val);
        if (!s) { PyErr_Clear(); return 0; }
        Py_ssize_t slen;
        const char *sdata = PyUnicode_AsUTF8AndSize(s, &slen);
        if (!sdata) { Py_DECREF(s); PyErr_Clear(); return 0; }
        h = fnv_bytes(h, (const uint8_t *)sdata, slen);
        Py_DECREF(s);
        return h;
    }

    /* Float */
    if (PyFloat_Check(val)) {
        h = fnv_byte(h, TAG_FLOAT);
        double d = PyFloat_AS_DOUBLE(val);
        uint64_t bits;
        if (d == 0.0) {
            bits = 0;
        } else if (isnan(d)) {
            bits = UINT64_C(0x7FF8000000000000);
        } else {
            /* Little-endian memcpy matches Python's struct.pack("<d") on LE systems.
               This is correct for x86, x86_64, ARM, AArch64 — all platforms
               where CPython runs in practice. */
            memcpy(&bits, &d, sizeof(bits));
        }
        return fnv_u64(h, bits);
    }

    /* String */
    if (PyUnicode_Check(val)) {
        h = fnv_byte(h, TAG_STR);
        Py_ssize_t len;
        const char *data = PyUnicode_AsUTF8AndSize(val, &len);
        if (!data) { PyErr_Clear(); return 0; }
        h = fnv_u64(h, (uint64_t)len);
        return fnv_bytes(h, (const uint8_t *)data, len);
    }

    /* Bytes */
    if (PyBytes_Check(val)) {
        h = fnv_byte(h, TAG_BYTES);
        Py_ssize_t len = PyBytes_GET_SIZE(val);
        h = fnv_u64(h, (uint64_t)len);
        return fnv_bytes(h, (const uint8_t *)PyBytes_AS_STRING(val), len);
    }

    /* List */
    if (PyList_Check(val)) {
        h = fnv_byte(h, TAG_LIST);
        Py_ssize_t n = PyList_GET_SIZE(val);
        h = fnv_u64(h, (uint64_t)n);
        for (Py_ssize_t i = 0; i < n; i++)
            h = fnv_u64(h, sh(PyList_GET_ITEM(val, i), depth + 1));
        return h;
    }

    /* Tuple */
    if (PyTuple_Check(val)) {
        h = fnv_byte(h, TAG_TUPLE);
        Py_ssize_t n = PyTuple_GET_SIZE(val);
        h = fnv_u64(h, (uint64_t)n);
        for (Py_ssize_t i = 0; i < n; i++)
            h = fnv_u64(h, sh(PyTuple_GET_ITEM(val, i), depth + 1));
        return h;
    }

    /* Dict — commutative XOR for order independence */
    if (PyDict_Check(val)) {
        h = fnv_byte(h, TAG_DICT);
        Py_ssize_t n = PyDict_Size(val);
        h = fnv_u64(h, (uint64_t)n);
        uint64_t acc = 0;
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(val, &pos, &key, &value)) {
            uint64_t kh = sh(key, depth + 1);
            uint64_t vh = sh(value, depth + 1);
            acc ^= kh ^ (vh * GOLDEN);
        }
        return fnv_u64(h, acc);
    }

    /* Set / FrozenSet — commutative XOR for order independence */
    if (PySet_Check(val) || PyFrozenSet_Check(val)) {
        h = fnv_byte(h, TAG_SET);
        Py_ssize_t n = PySet_GET_SIZE(val);
        h = fnv_u64(h, (uint64_t)n);
        uint64_t acc = 0;
        PyObject *iter = PyObject_GetIter(val);
        if (!iter) { PyErr_Clear(); return fnv_u64(h, 0); }
        PyObject *item;
        while ((item = PyIter_Next(iter)) != NULL) {
            acc ^= sh(item, depth + 1);
            Py_DECREF(item);
        }
        Py_DECREF(iter);
        if (PyErr_Occurred()) PyErr_Clear();
        return fnv_u64(h, acc);
    }

    /* ---- Slow path: Pydantic models, dataclasses, arbitrary objects ---- */

    /* Pydantic model_dump() */
    if (PyObject_HasAttrString(val, "model_dump")) {
        PyObject *dumped = PyObject_CallMethod(val, "model_dump", NULL);
        if (dumped) {
            uint64_t r = sh(dumped, depth + 1);
            Py_DECREF(dumped);
            return r;
        }
        PyErr_Clear();
    }

    /* Dataclass — dataclasses.asdict(val) */
    if (PyObject_HasAttrString(val, "__dataclass_fields__")) {
        PyObject *dc_mod = PyImport_ImportModule("dataclasses");
        if (dc_mod) {
            PyObject *asdict_fn = PyObject_GetAttrString(dc_mod, "asdict");
            Py_DECREF(dc_mod);
            if (asdict_fn) {
                PyObject *d = PyObject_CallOneArg(asdict_fn, val);
                Py_DECREF(asdict_fn);
                if (d) {
                    uint64_t r = sh(d, depth + 1);
                    Py_DECREF(d);
                    return r;
                }
            }
        }
        PyErr_Clear();
    }

    /* Fallback: __dict__ */
    if (PyObject_HasAttrString(val, "__dict__")) {
        PyObject *d = PyObject_GetAttrString(val, "__dict__");
        if (d) {
            uint64_t r = sh(d, depth + 1);
            Py_DECREF(d);
            return r;
        }
        PyErr_Clear();
    }

    return 0;
}

/* ========================================================================= */
/* Output derivation — splitmix64 + hex encoding                             */
/* ========================================================================= */

static inline uint64_t splitmix64(uint64_t z)
{
    z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
    return z ^ (z >> 31);
}

static void make_hex(uint64_t h, char *buf, int length)
{
    static const char hex[] = "0123456789abcdef";
    uint64_t vals[4];
    vals[0] = h;
    vals[1] = splitmix64(h + GOLDEN * 1);
    vals[2] = splitmix64(h + GOLDEN * 2);
    vals[3] = splitmix64(h + GOLDEN * 3);

    int pos = 0;
    for (int vi = 0; vi < 4 && pos < length; vi++) {
        uint64_t v = vals[vi];
        for (int bi = 60; bi >= 0 && pos < length; bi -= 4)
            buf[pos++] = hex[(v >> bi) & 0xF];
    }
    buf[pos] = '\0';
}

/* ========================================================================= */
/* Python-exposed function                                                   */
/* ========================================================================= */

PyDoc_STRVAR(fast_hash_doc,
"fast_hash(obj, length=16) -> str\n\n"
"High-performance structural FNV-1a hash.\n"
"Hashes the object tree directly without serialization.\n"
"Dicts/sets use commutative XOR — insertion order never matters.\n"
"Produces identical output to the pure-Python fallback.\n\n"
"Args:\n"
"    obj: Any Python object\n"
"    length: Hex output length (1-64, default 16)\n\n"
"Returns:\n"
"    Hex hash string\n");

static PyObject *
py_fast_hash(PyObject *self, PyObject *args)
{
    PyObject *obj;
    int length = 16;

    if (!PyArg_ParseTuple(args, "O|i", &obj, &length))
        return NULL;

    if (length < 1)  length = 1;
    if (length > 64) length = 64;

    uint64_t h = sh(obj, 0);
    char buf[65];
    make_hex(h, buf, length);

    return PyUnicode_FromStringAndSize(buf, length);
}

/* ========================================================================= */
/* Module definition                                                         */
/* ========================================================================= */

static PyMethodDef module_methods[] = {
    {"fast_hash", py_fast_hash, METH_VARARGS, fast_hash_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_native_hash",                              /* m_name */
    "C-accelerated structural hashing for stampede.\n\n"
    "Provides fast_hash() — a 10-50x faster replacement for\n"
    "the pure-Python FNV-1a structural hash.",   /* m_doc */
    -1,                                          /* m_size */
    module_methods                               /* m_methods */
};

PyMODINIT_FUNC
PyInit__native_hash(void)
{
    return PyModule_Create(&moduledef);
}
