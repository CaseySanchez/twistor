#include <memory>
#include <vector>
#include <string>
#include <iostream>

#include <H5Cpp.h>

#include <thrust/host_vector.h>

#include "twistor/thrust/cuda/stream.hpp"
#include "twistor/thrust/async_device_vector.hpp"
#include "twistor/gatl/csta.hpp"
#include "twistor/holonomy.hpp"

struct GaugeFieldFunctor
{
    /**
     * @brief The gauge field consists of a globally parabolic space and a locally hyperbolic space near a perturbation.
     */
    __device__ vector_t operator()(vector_t const &vector) const
    {
        return global_infinity_field(vector) + perturbation_field(vector);
    }

private:
    /**
     * @brief The globally parabolic space.
     */
    __device__ vector_t global_infinity_field(vector_t const &vector) const
    {
        return c<0> * et + c<0> * ex + c<0> * ey + c<0> * ez + ni;
    }

    /**
     * @brief The locally hyperbolic space.
     */
    __device__ vector_t perturbation_field(vector_t const &vector) const
    {
        auto const perturbation_position = 0.5 * et + 0.5 * ex + 0.5 * ey + 0.5 * ez + c<0> * ep + c<0> * em;
        auto const perturbation_field_strength = c<0> * et + c<0> * ex + c<0> * ey + c<0> * ez + -0.03 * ep + 0.03 * em;

        auto const difference = vector - perturbation_position;
        float const weight = 1.0 / ((difference | difference) + 1e-2);

        return weight * perturbation_field_strength;
    }
};

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);

    if (args.size() != 1) {
        std::cout << "Usage: ./holonomy {OUTPUT FILE}" << std::endl;

        return 1;
    }

    // Create a CUDA stream

    std::shared_ptr<thrust::cuda::stream> stream = std::make_shared<thrust::cuda::stream>();

    // Compute the holonomy given the GaugeFieldFunctor

    size_t const t_size = 20;
    size_t const x_size = 20;
    size_t const y_size = 20;
    size_t const z_size = 20;

    float const t_min = -1.0;
    float const x_min = -1.0;
    float const y_min = -1.0;
    float const z_min = -1.0;

    float const t_max = 1.0;
    float const x_max = 1.0;
    float const y_max = 1.0;
    float const z_max = 1.0;

    Holonomy holonomy(stream, t_size, x_size, y_size, z_size, t_min, x_min, y_min, z_min, t_max, x_max, y_max, z_max);

    if (holonomy.compute(GaugeFieldFunctor{}).info() != ComputationInfo::Success) {
        std::cout << "Failed to compute the holonomy." << std::endl;

        return 1;
    }

    // Synchronize the stream

    stream->synchronize();

    // HDF5 file output

    std::string const &filename = args[0];

    H5::H5File file(filename, H5F_ACC_TRUNC);

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_tx().begin(), holonomy.holonomy_tx().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_tx", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_xy().begin(), holonomy.holonomy_xy().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_xy", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_yz().begin(), holonomy.holonomy_yz().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_yz", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_zt().begin(), holonomy.holonomy_zt().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_zt", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_ty().begin(), holonomy.holonomy_ty().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_ty", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    {
        thrust::host_vector<motor_t> host_vector(holonomy.holonomy_xz().begin(), holonomy.holonomy_xz().end());

        std::array<hsize_t, 2> const dims = {
            static_cast<hsize_t>(host_vector.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const dataspace(dims.size(), dims.data());

        file.createDataSet("holonomy_xz", H5::PredType::NATIVE_FLOAT, dataspace).write(host_vector.data(), H5::PredType::NATIVE_FLOAT);
    }

    std::cout << "Holonomy data written to: " << filename << std::endl;

    return 0;
}
