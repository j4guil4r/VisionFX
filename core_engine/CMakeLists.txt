include(FetchContent)

# Descargar pybind11 al momento de compilar
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# Compilar el wrapper como un módulo de Python (.so)
pybind11_add_module(ar_helmet_core bindings/pybind_wrapper.cpp)