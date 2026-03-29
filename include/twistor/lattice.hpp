#pragma once

#include <cstddef>

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

    size_t t_size() const
    {
        return m_t_size;
    }

    size_t x_size() const
    {
        return m_x_size;
    }

    size_t y_size() const
    {
        return m_y_size;
    }

    size_t z_size() const
    {
        return m_z_size;
    }

    float t_min() const
    {
        return m_t_min;
    }

    float x_min() const
    {
        return m_x_min;
    }

    float y_min() const
    {
        return m_y_min;
    }

    float z_min() const
    {
        return m_z_min;
    }

    float t_max() const
    {
        return m_t_max;
    }

    float x_max() const
    {
        return m_x_max;
    }

    float y_max() const
    {
        return m_y_max;
    }

    float z_max() const
    {
        return m_z_max;
    }

    float t_step() const
    {
        return (m_t_max - m_t_min) / static_cast<float>(m_t_size);
    }

    float x_step() const
    {
        return (m_x_max - m_x_min) / static_cast<float>(m_x_size);
    }

    float y_step() const
    {
        return (m_y_max - m_y_min) / static_cast<float>(m_y_size);
    }

    float z_step() const
    {
        return (m_z_max - m_z_min) / static_cast<float>(m_z_size);
    }
};