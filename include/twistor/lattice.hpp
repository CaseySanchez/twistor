/**
 * @file lattice.hpp
 * @brief Parameterizes a lattice structure.
 */

#pragma once

#include <cstddef>

#ifdef TWISTOR_HDF5
#include <H5Cpp.h>
#endif

namespace Twistor
{
    class Lattice;
}

#ifdef TWISTOR_HDF5
namespace Twistor::HDF5
{
    void read(H5::Group const &group, Twistor::Lattice &lattice);
    void write(H5::Group &group, Twistor::Lattice const &lattice);
}
#endif

namespace Twistor
{
    class Lattice
    {
        size_t m_t_size;
        size_t m_x_size;
        size_t m_y_size;
        size_t m_z_size;

        float m_t_min;
        float m_x_min;
        float m_y_min;
        float m_z_min;

        float m_t_max;
        float m_x_max;
        float m_y_max;
        float m_z_max;

    public:
        /**
         * @brief Default constructor.
         */
        Lattice() = default;

        /**
         * @brief Constructs a lattice object.
         * @param [in] t_size Number of lattice sites along the t-axis.
         * @param [in] x_size Number of lattice sites along the x-axis.
         * @param [in] y_size Number of lattice sites along the y-axis.
         * @param [in] z_size Number of lattice sites along the z-axis.
         * @param [in] t_min Physical lower bound of the t-axis.
         * @param [in] x_min Physical lower bound of the x-axis.
         * @param [in] y_min Physical lower bound of the y-axis.
         * @param [in] z_min Physical lower bound of the z-axis.
         * @param [in] t_max Physical upper bound of the t-axis.
         * @param [in] x_max Physical upper bound of the x-axis.
         * @param [in] y_max Physical upper bound of the y-axis.
         * @param [in] z_max Physical upper bound of the z-axis.
         */
        Lattice(
            size_t const t_size,
            size_t const x_size,
            size_t const y_size,
            size_t const z_size,
            float const t_min,
            float const x_min,
            float const y_min,
            float const z_min,
            float const t_max,
            float const x_max,
            float const y_max,
            float const z_max);

        size_t t_size() const;

        size_t x_size() const;

        size_t y_size() const;

        size_t z_size() const;

        float t_min() const;

        float x_min() const;

        float y_min() const;

        float z_min() const;

        float t_max() const;

        float x_max() const;

        float y_max() const;

        float z_max() const;

        float t_step() const;

        float x_step() const;

        float y_step() const;

        float z_step() const;

#ifdef TWISTOR_HDF5
    private:
        friend void Twistor::HDF5::read(H5::Group const &group, Twistor::Lattice &lattice);
        friend void Twistor::HDF5::write(H5::Group &group, Twistor::Lattice const &lattice);
#endif
    };
}

inline Twistor::Lattice::Lattice(
    size_t const t_size,
    size_t const x_size,
    size_t const y_size,
    size_t const z_size,
    float const t_min,
    float const x_min,
    float const y_min,
    float const z_min,
    float const t_max,
    float const x_max,
    float const y_max,
    float const z_max) :
    m_t_size { t_size },
    m_x_size { x_size },
    m_y_size { y_size },
    m_z_size { z_size },
    m_t_min { t_min },
    m_x_min { x_min },
    m_y_min { y_min },
    m_z_min { z_min },
    m_t_max { t_max },
    m_x_max { x_max },
    m_y_max { y_max },
    m_z_max { z_max }
{
}

size_t Twistor::Lattice::t_size() const
{
    return m_t_size;
}

size_t Twistor::Lattice::x_size() const
{
    return m_x_size;
}

size_t Twistor::Lattice::y_size() const
{
    return m_y_size;
}

size_t Twistor::Lattice::z_size() const
{
    return m_z_size;
}

float Twistor::Lattice::t_min() const
{
    return m_t_min;
}

float Twistor::Lattice::x_min() const
{
    return m_x_min;
}

float Twistor::Lattice::y_min() const
{
    return m_y_min;
}

float Twistor::Lattice::z_min() const
{
    return m_z_min;
}

float Twistor::Lattice::t_max() const
{
    return m_t_max;
}

float Twistor::Lattice::x_max() const
{
    return m_x_max;
}

float Twistor::Lattice::y_max() const
{
    return m_y_max;
}

float Twistor::Lattice::z_max() const
{
    return m_z_max;
}

float Twistor::Lattice::t_step() const
{
    return (m_t_max - m_t_min) / static_cast<float>(m_t_size);
}

float Twistor::Lattice::x_step() const
{
    return (m_x_max - m_x_min) / static_cast<float>(m_x_size);
}

float Twistor::Lattice::y_step() const
{
    return (m_y_max - m_y_min) / static_cast<float>(m_y_size);
}

float Twistor::Lattice::z_step() const
{
    return (m_z_max - m_z_min) / static_cast<float>(m_z_size);
}

#ifdef TWISTOR_HDF5
namespace Twistor::HDF5
{
    void read(H5::Group const &group, Twistor::Lattice &lattice)
    {
        group.openAttribute("t_size").read(H5::PredType::NATIVE_HSIZE, &lattice.m_t_size);
        group.openAttribute("x_size").read(H5::PredType::NATIVE_HSIZE, &lattice.m_x_size);
        group.openAttribute("y_size").read(H5::PredType::NATIVE_HSIZE, &lattice.m_y_size);
        group.openAttribute("z_size").read(H5::PredType::NATIVE_HSIZE, &lattice.m_z_size);

        group.openAttribute("t_min").read(H5::PredType::NATIVE_FLOAT, &lattice.m_t_min);
        group.openAttribute("x_min").read(H5::PredType::NATIVE_FLOAT, &lattice.m_x_min);
        group.openAttribute("y_min").read(H5::PredType::NATIVE_FLOAT, &lattice.m_y_min);
        group.openAttribute("z_min").read(H5::PredType::NATIVE_FLOAT, &lattice.m_z_min);

        group.openAttribute("t_max").read(H5::PredType::NATIVE_FLOAT, &lattice.m_t_max);
        group.openAttribute("x_max").read(H5::PredType::NATIVE_FLOAT, &lattice.m_x_max);
        group.openAttribute("y_max").read(H5::PredType::NATIVE_FLOAT, &lattice.m_y_max);
        group.openAttribute("z_max").read(H5::PredType::NATIVE_FLOAT, &lattice.m_z_max);
    }

    void write(H5::Group &group, Twistor::Lattice const &lattice)
    {
        H5::DataSpace const dataspace(H5S_SCALAR);

        group.createAttribute("t_size", H5::PredType::NATIVE_HSIZE, dataspace).write(H5::PredType::NATIVE_HSIZE, &lattice.m_t_size);
        group.createAttribute("x_size", H5::PredType::NATIVE_HSIZE, dataspace).write(H5::PredType::NATIVE_HSIZE, &lattice.m_x_size);
        group.createAttribute("y_size", H5::PredType::NATIVE_HSIZE, dataspace).write(H5::PredType::NATIVE_HSIZE, &lattice.m_y_size);
        group.createAttribute("z_size", H5::PredType::NATIVE_HSIZE, dataspace).write(H5::PredType::NATIVE_HSIZE, &lattice.m_z_size);

        group.createAttribute("t_min", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_t_min);
        group.createAttribute("x_min", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_x_min);
        group.createAttribute("y_min", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_y_min);
        group.createAttribute("z_min", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_z_min);

        group.createAttribute("t_max", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_t_max);
        group.createAttribute("x_max", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_x_max);
        group.createAttribute("y_max", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_y_max);
        group.createAttribute("z_max", H5::PredType::NATIVE_FLOAT, dataspace).write(H5::PredType::NATIVE_FLOAT, &lattice.m_z_max);
    }
}
#endif