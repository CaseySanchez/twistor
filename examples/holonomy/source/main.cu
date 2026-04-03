#include <vector>
#include <string>
#include <iostream>

#include <H5Cpp.h>

#include "twistor/csta.hpp"
#include "twistor/holonomy.hpp"

struct GaugeFieldFunctor
{
    /**
     * @brief The gauge field consists of a globally parabolic space and a locally hyperbolic space near a perturbation.
     */
    __device__ vector_t operator()(vector_t vector) const
    {
        return global_infinity_field(vector) + perturbation_field(vector);
    }

private:
    /**
     * @brief The globally parabolic space.
     */
    __device__ vector_t global_infinity_field(vector_t vector) const
    {
        return c<0> * et + c<0> * ex + c<0> * ey + c<0> * ez + ni;
    }

    /**
     * @brief The locally hyperbolic space.
     */
    __device__ vector_t perturbation_field(vector_t vector) const
    {
        auto const perturbation_position = 0.5 * et + 0.5 * ex + 0.5 * ey + 0.5 * ez + c<0> * ep + c<0> * em;
        auto const perturbation_field_strength = c<0> * et + c<0> * ex + c<0> * ey + c<0> * ez + -0.01 * ep + 0.01 * em;

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

    // Create a lattice object

    size_t const t_size = 10;
    size_t const x_size = 50;
    size_t const y_size = 50;
    size_t const z_size = 50;

    float const t_min = -1.0;
    float const x_min = -1.0;
    float const y_min = -1.0;
    float const z_min = -1.0;

    float const t_max = 1.0;
    float const x_max = 1.0;
    float const y_max = 1.0;
    float const z_max = 1.0;

    Twistor::Lattice lattice(t_size, x_size, y_size, z_size, t_min, x_min, y_min, z_min, t_max, x_max, y_max, z_max);

    // Create a device holonomy object

    Twistor::Device::Holonomy device_holonomy(lattice);

    // Compute the holonomy given the GaugeFieldFunctor

    if (device_holonomy.compute(GaugeFieldFunctor{}).info() != Twistor::ComputationInfo::Success) {
        std::cout << "Failed to compute the holonomy." << std::endl;

        return 1;
    }

    // Copy holonomy from device to host

    Twistor::Host::Holonomy host_holonomy(device_holonomy);

    // HDF5 file output

    std::string const &filename = args[0];

    H5::H5File file(filename, H5F_ACC_TRUNC);

    H5::Group lattice_group = file.createGroup("lattice");
    H5::Group holonomy_group = file.createGroup("holonomy");

    Twistor::HDF5::write(lattice_group, lattice);
    Twistor::HDF5::write(holonomy_group, host_holonomy);

    std::cout << "Holonomy data written to: " << filename << std::endl;

    return 0;
}
