#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include "tensor.hpp"
#include "cpu_kernels.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_nano_infer, m) {
    m.doc() = "NanoInfer: A High-Performance CUDA Inference Engine";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::enum_<DType>(m, "DType")
        .value("Int32", DType::Int32)
        .value("Float32", DType::Float32)
        .export_values();

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int>, DType, Device>(), 
                py::arg("shape"), 
                py::arg("dtype"), 
                py::arg("device"))
        .def(py::init([](py::array b, Device device) {
            py::buffer_info info = b.request();
            
            std::vector<int> shape;
            for (auto s : info.shape) shape.push_back(static_cast<int>(s));

            DType dtype = DType::Float32;
            if (info.format == py::format_descriptor<float>::format()) {
                dtype = DType::Float32;
            } else if (info.format == py::format_descriptor<int>::format() || info.format == "i") {
                dtype = DType::Int32;
            } else {
                throw std::runtime_error("Unsupported dtype: " + info.format);
            }

            Tensor* t = nullptr;

            if (device == Device::CPU) {
                t = new Tensor(shape, dtype, Device::CPU);
                std::memcpy(t->data, info.ptr, t->nbytes());
            } else {
                Tensor* temp_cpu = new Tensor(shape, dtype, Device::CPU);
                std::memcpy(temp_cpu->data, info.ptr, temp_cpu->nbytes());
                temp_cpu->to_cuda(); 
                t = temp_cpu;
            }
            return t;
        }), py::arg("array"), py::arg("device") = Device::CPU)
        
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readonly("size", &Tensor::size)
        .def_readonly("device", &Tensor::device)
        .def_readonly("dtype", &Tensor::dtype)

        .def_property_readonly("data_ptr", [](Tensor& t) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(t.data);
        })
        
        .def_buffer([](Tensor &t) -> py::buffer_info {
            if (t.device == Device::CUDA) {
                throw std::runtime_error("Cannot directly access CUDA tensor data via Buffer Protocol. Use .to_cpu() first.");
            }

            std::vector<ssize_t> strides_in_bytes;
            size_t elem_size = t.element_size();

            for (int s : t.strides) {
                strides_in_bytes.push_back(s * elem_size);
            }
            std::vector<ssize_t> shape_ssize_t(t.shape.begin(), t.shape.end());

            std::string format;
            if (t.dtype == DType::Float32) format = py::format_descriptor<float>::format(); // "f"
            else format = py::format_descriptor<int>::format(); // "i"

            return py::buffer_info(
                t.data,
                elem_size,
                format,
                t.shape.size(),
                shape_ssize_t,
                strides_in_bytes
            );
        })

        .def("to_cuda", &Tensor::to_cuda, py::return_value_policy::reference)
        .def("to_cpu", &Tensor::to_cpu, py::return_value_policy::reference)

        .def("add", &Tensor::add, py::return_value_policy::take_ownership)
        .def("mul", &Tensor::mul, py::return_value_policy::take_ownership)
        .def("matmul", &Tensor::matmul, 
            py::arg("other"), 
            py::arg("trans_a") = false, 
            py::arg("trans_b") = false,
            py::return_value_policy::take_ownership)
        
        .def("__repr__", &Tensor::to_string);

        m.def("silu", &silu, "SiLU CPU Naive Implementation");
        m.def("embedding", &embedding, "Embedding CPU Naive Implementation");
        m.def("softmax", &softmax, "Softmax CPU Naive Implementation");
        m.def("rope", &rope, "RoPE CPU Naive Implementation (In-place)");
}