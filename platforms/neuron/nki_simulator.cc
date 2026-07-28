#include "platforms/neuron/nki_simulator.h"

#include <Python.h>

#include <limits>
#include <mutex>
#include <string>

namespace {

std::mutex initialize_mutex;
std::mutex execute_mutex;
PyObject* run_function = nullptr;
thread_local std::string last_error;

const char* SetError(const char* prefix) {
  last_error = prefix;
  if (!PyErr_Occurred()) return last_error.c_str();

  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  if (value != nullptr) {
    PyObject* rendered = PyObject_Str(value);
    if (rendered != nullptr) {
      const char* text = PyUnicode_AsUTF8(rendered);
      if (text != nullptr) {
        last_error.append(": ");
        last_error.append(text);
      }
      Py_DECREF(rendered);
    }
  }
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  return last_error.c_str();
}

bool CheckStatus(PyStatus status, const char* operation) {
  if (!PyStatus_Exception(status)) return true;
  last_error = operation;
  if (status.err_msg != nullptr) {
    last_error.append(": ");
    last_error.append(status.err_msg);
  }
  return false;
}

bool AppendModulePath(PyConfig* config, const char* path) {
  wchar_t* wide_path = Py_DecodeLocale(path, nullptr);
  if (wide_path == nullptr) {
    last_error = "failed to decode embedded Python module path";
    return false;
  }
  const PyStatus status =
      PyWideStringList_Append(&config->module_search_paths, wide_path);
  PyMem_RawFree(wide_path);
  return CheckStatus(status, "failed to append embedded Python module path");
}

PyObject* BufferList(const zml_nki_simulator_buffer* buffers, size_t count) {
  PyObject* list = PyList_New(static_cast<Py_ssize_t>(count));
  if (list == nullptr) return nullptr;

  for (size_t i = 0; i < count; ++i) {
    const zml_nki_simulator_buffer& buffer = buffers[i];
    if (buffer.byte_size >
        static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max())) {
      Py_DECREF(list);
      PyErr_SetString(PyExc_OverflowError,
                      "NKI simulator buffer is too large for Python");
      return nullptr;
    }

    PyObject* view = PyMemoryView_FromMemory(
        static_cast<char*>(buffer.data),
        static_cast<Py_ssize_t>(buffer.byte_size), PyBUF_WRITE);
    PyObject* dtype =
        PyUnicode_FromStringAndSize(buffer.dtype, buffer.dtype_len);
    PyObject* shape = PyTuple_New(static_cast<Py_ssize_t>(buffer.rank));
    if (view == nullptr || dtype == nullptr || shape == nullptr) {
      Py_XDECREF(view);
      Py_XDECREF(dtype);
      Py_XDECREF(shape);
      Py_DECREF(list);
      return nullptr;
    }
    for (size_t dim = 0; dim < buffer.rank; ++dim) {
      PyObject* value = PyLong_FromLongLong(buffer.dims[dim]);
      if (value == nullptr) {
        Py_DECREF(view);
        Py_DECREF(dtype);
        Py_DECREF(shape);
        Py_DECREF(list);
        return nullptr;
      }
      PyTuple_SET_ITEM(shape, static_cast<Py_ssize_t>(dim), value);
    }

    PyObject* descriptor = PyTuple_Pack(3, view, dtype, shape);
    Py_DECREF(view);
    Py_DECREF(dtype);
    Py_DECREF(shape);
    if (descriptor == nullptr) {
      Py_DECREF(list);
      return nullptr;
    }
    PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), descriptor);
  }
  return list;
}

}  // namespace

extern "C" const char* zml_nki_simulator_initialize(
    const char* python_home, const char* site_packages,
    const char* bridge_directory) {
  std::lock_guard<std::mutex> lock(initialize_mutex);
  if (run_function != nullptr) return nullptr;
  if (Py_IsInitialized()) {
    last_error =
        "cannot initialize the NKI simulator after another embedded Python "
        "interpreter";
    return last_error.c_str();
  }

  PyPreConfig preconfig;
  PyPreConfig_InitIsolatedConfig(&preconfig);
  preconfig.utf8_mode = 1;
  if (!CheckStatus(Py_PreInitialize(&preconfig),
                   "failed to preinitialize embedded Python")) {
    return last_error.c_str();
  }

  PyConfig config;
  PyConfig_InitIsolatedConfig(&config);
  config.module_search_paths_set = 1;
  config.optimization_level = 2;
  config.write_bytecode = 0;

  if (!CheckStatus(PyConfig_SetBytesString(&config, &config.home, python_home),
                   "failed to configure embedded Python home") ||
      !AppendModulePath(&config, python_home) ||
      !AppendModulePath(&config, site_packages) ||
      !AppendModulePath(&config, bridge_directory) ||
      !CheckStatus(Py_InitializeFromConfig(&config),
                   "failed to initialize embedded Python")) {
    PyConfig_Clear(&config);
    return last_error.c_str();
  }
  PyConfig_Clear(&config);

  PyObject* module = PyImport_ImportModule("nki_simulator_bridge");
  if (module == nullptr) return SetError("failed to import NKI simulator bridge");
  run_function = PyObject_GetAttrString(module, "run");
  Py_DECREF(module);
  if (run_function == nullptr || !PyCallable_Check(run_function)) {
    Py_XDECREF(run_function);
    run_function = nullptr;
    return SetError("NKI simulator bridge has no callable run function");
  }

  // PJRT invokes FFI handlers from worker threads. Release the initialization
  // thread's GIL so each callback can acquire it with PyGILState_Ensure.
  PyEval_SaveThread();
  return nullptr;
}

extern "C" const char* zml_nki_simulator_execute(
    const char* source, size_t source_len, const char* entrypoint,
    size_t entrypoint_len, const char* compiler_target,
    size_t compiler_target_len, int64_t grid,
    const zml_nki_simulator_buffer* inputs, size_t input_count,
    const zml_nki_simulator_buffer* outputs, size_t output_count) {
  // NKI's simulator backend is process-global. NumPy operations can release
  // the GIL, so the GIL alone does not prevent another PJRT worker from
  // replacing the active backend halfway through a launch.
  std::lock_guard<std::mutex> lock(execute_mutex);
  if (run_function == nullptr) {
    last_error = "NKI simulator is not initialized";
    return last_error.c_str();
  }

  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* source_object =
      PyUnicode_FromStringAndSize(source, static_cast<Py_ssize_t>(source_len));
  PyObject* entrypoint_object = PyUnicode_FromStringAndSize(
      entrypoint, static_cast<Py_ssize_t>(entrypoint_len));
  PyObject* target_object = PyUnicode_FromStringAndSize(
      compiler_target, static_cast<Py_ssize_t>(compiler_target_len));
  PyObject* grid_object = PyLong_FromLongLong(grid);
  PyObject* input_list = BufferList(inputs, input_count);
  PyObject* output_list = BufferList(outputs, output_count);

  if (source_object == nullptr || entrypoint_object == nullptr ||
      target_object == nullptr || grid_object == nullptr ||
      input_list == nullptr || output_list == nullptr) {
    Py_XDECREF(source_object);
    Py_XDECREF(entrypoint_object);
    Py_XDECREF(target_object);
    Py_XDECREF(grid_object);
    Py_XDECREF(input_list);
    Py_XDECREF(output_list);
    const char* error = SetError("failed to prepare NKI simulator arguments");
    PyGILState_Release(gil);
    return error;
  }

  PyObject* result = PyObject_CallFunctionObjArgs(
      run_function, source_object, entrypoint_object, target_object, grid_object,
      input_list, output_list, nullptr);
  Py_DECREF(source_object);
  Py_DECREF(entrypoint_object);
  Py_DECREF(target_object);
  Py_DECREF(grid_object);
  Py_DECREF(input_list);
  Py_DECREF(output_list);

  if (result == nullptr) {
    const char* error = SetError("NKI simulator execution failed");
    PyGILState_Release(gil);
    return error;
  }
  Py_DECREF(result);
  PyGILState_Release(gil);
  return nullptr;
}
