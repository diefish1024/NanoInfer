#include <pybind11/pybind11.h>
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

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<std::vector<int>, Device>(), 
             py::arg("shape"), py::arg("device") = Device::CUDA)
        
        .def_readonly("shape", &Tensor::shape)
        .def_readonly("strides", &Tensor::strides)
        .def_readonly("size", &Tensor::size)
        .def_readonly("device", &Tensor::device)
        .def_property_readonly("data_ptr", [](Tensor& t) -> std::uintptr_t {
            return reinterpret_cast<std::uintptr_t>(t.data);
        })

        .def_buffer([](Tensor &t) -> py::buffer_info {
            if (t.device == Device::CUDA) {
                throw std::runtime_error("Cannot directly access CUDA tensor data via Buffer Protocol. Use .to_cpu() first.");
            }

            std::vector<ssize_t> strides_in_bytes;
            for (int s : t.strides) {
                strides_in_bytes.push_back(s * sizeof(float));
            }
            
            std::vector<ssize_t> shape_ssize_t(t.shape.begin(), t.shape.end());

            return py::buffer_info(
                t.data,
                sizeof(float),
                py::format_descriptor<float>::format(),
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
}