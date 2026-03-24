/**
 * @file holonomy.hpp
 * @author Casey Sanchez
 * @brief Class to compute the holonomy on a lattice.
 */

#pragma once

#include <memory>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "twistor/thrust/cuda/stream.hpp"
#include "twistor/thrust/async_device_vector.hpp"
#include "twistor/gatl/csta.hpp"
#include "twistor/computation_info.hpp"

template <typename GaugeFieldFunctor>
concept GaugeFieldFunctorConcept = requires(GaugeFieldFunctor const &gauge_field_functor, vector_t const &vector) {
    { gauge_field_functor(vector) } -> std::same_as<vector_t>;
};

class Holonomy
{
    std::shared_ptr<thrust::cuda::stream> m_stream;

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

    ComputationInfo m_info;

    thrust::async_device_vector<motor_t> m_holonomy_tx;
    thrust::async_device_vector<motor_t> m_holonomy_xy;
    thrust::async_device_vector<motor_t> m_holonomy_yz;
    thrust::async_device_vector<motor_t> m_holonomy_zt;
    thrust::async_device_vector<motor_t> m_holonomy_ty;
    thrust::async_device_vector<motor_t> m_holonomy_xz;

public:
    /**
     * @brief Constructs a Holonomy object for the given lattice dimensions and physical domain.
     * @param [in] stream CUDA stream used for all device allocations and kernel launches.
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
    Holonomy(
        std::shared_ptr<thrust::cuda::stream> const &stream,
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
        m_stream { stream },
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
        m_z_max { z_max },
        m_info { ComputationInfo::Success },
        m_holonomy_tx ( thrust::cuda::async_allocator<motor_t> { *stream } ),
        m_holonomy_xy ( thrust::cuda::async_allocator<motor_t> { *stream } ),
        m_holonomy_yz ( thrust::cuda::async_allocator<motor_t> { *stream } ),
        m_holonomy_zt ( thrust::cuda::async_allocator<motor_t> { *stream } ),
        m_holonomy_ty ( thrust::cuda::async_allocator<motor_t> { *stream } ),
        m_holonomy_xz ( thrust::cuda::async_allocator<motor_t> { *stream } )
    {
    }

    /** 
     * @brief Returns the status of the most recent call to `Holonomy::compute`.
     */
    ComputationInfo info() const
    {
        return m_info;
    }

    /** 
     * @brief Device buffer of tx-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_tx() const
    {
        return m_holonomy_tx;
    }

    /** 
     * @brief Device buffer of xy-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_xy() const
    {
        return m_holonomy_xy;
    }

    /** 
     * @brief Device buffer of yz-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_yz() const
    {
        return m_holonomy_yz;
    }

    /** 
     * @brief Device buffer of zt-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_zt() const
    {
        return m_holonomy_zt;
    }

    /** 
     * @brief Device buffer of ty-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_ty() const
    {
        return m_holonomy_ty;
    }

    /** 
     * @brief Device buffer of xz-plane plaquette holonomies.
     */
    thrust::async_device_vector<motor_t> const &holonomy_xz() const
    {
        return m_holonomy_xz;
    }

    /**
     * @brief Schedules the holonomy kernel on the CUDA stream.
     * @param [in] gauge_field_functor Device-callable functor satisfying `GaugeFieldFunctorConcept`.
     * @return Reference to `this`.
     */
    template <GaugeFieldFunctorConcept GaugeFieldFunctor>
    Holonomy &compute(GaugeFieldFunctor gauge_field_functor)
    {
        size_t const t_size = m_t_size;
        size_t const x_size = m_x_size;
        size_t const y_size = m_y_size;
        size_t const z_size = m_z_size;

        float const t_min = m_t_min;
        float const x_min = m_x_min;
        float const y_min = m_y_min;
        float const z_min = m_z_min;

        float const t_max = m_t_max;
        float const x_max = m_x_max;
        float const y_max = m_y_max;
        float const z_max = m_z_max;

        float const t_step = (t_max - t_min) / static_cast<float>(t_size);
        float const x_step = (x_max - x_min) / static_cast<float>(x_size);
        float const y_step = (y_max - y_min) / static_cast<float>(y_size);
        float const z_step = (z_max - z_min) / static_cast<float>(z_size);

        size_t const t_stride = x_size * y_size * z_size;
        size_t const x_stride = y_size * z_size;
        size_t const y_stride = z_size;

        auto index_iterator = thrust::make_counting_iterator<size_t>(0);

        auto t_index_iterator = thrust::make_transform_iterator(
            index_iterator,
            [t_stride] __device__ (size_t const &index) -> size_t {
                return index / t_stride;
            }
        );

        auto x_index_iterator = thrust::make_transform_iterator(
            index_iterator,
            [x_stride, x_size] __device__ (size_t const &index) -> size_t {
                return (index / x_stride) % x_size;
            }
        );

        auto y_index_iterator = thrust::make_transform_iterator(
            index_iterator,
            [y_stride, y_size] __device__ (size_t const &index) -> size_t {
                return (index / y_stride) % y_size;
            }
        );

        auto z_index_iterator = thrust::make_transform_iterator(
            index_iterator,
            [z_size] __device__ (size_t const &index) -> size_t {
                return index % z_size;
            }
        );

        auto index_iterator_begin = thrust::make_zip_iterator(thrust::make_tuple(index_iterator, t_index_iterator, x_index_iterator, y_index_iterator, z_index_iterator));
        auto index_iterator_end = thrust::next(index_iterator_begin, t_size * x_size * y_size * z_size);

        m_holonomy_tx.resize(t_size * x_size * y_size * z_size);
        m_holonomy_xy.resize(t_size * x_size * y_size * z_size);
        m_holonomy_yz.resize(t_size * x_size * y_size * z_size);
        m_holonomy_zt.resize(t_size * x_size * y_size * z_size);
        m_holonomy_ty.resize(t_size * x_size * y_size * z_size);
        m_holonomy_xz.resize(t_size * x_size * y_size * z_size);

        auto holonomy_tx_ptr = thrust::raw_pointer_cast(m_holonomy_tx.data());
        auto holonomy_xy_ptr = thrust::raw_pointer_cast(m_holonomy_xy.data());
        auto holonomy_yz_ptr = thrust::raw_pointer_cast(m_holonomy_yz.data());
        auto holonomy_zt_ptr = thrust::raw_pointer_cast(m_holonomy_zt.data());
        auto holonomy_ty_ptr = thrust::raw_pointer_cast(m_holonomy_ty.data());
        auto holonomy_xz_ptr = thrust::raw_pointer_cast(m_holonomy_xz.data());

        thrust::for_each(
            thrust::cuda::par.on(*m_stream),
            index_iterator_begin,
            index_iterator_end,
            [
                gauge_field_functor,
                holonomy_tx_ptr,
                holonomy_xy_ptr,
                holonomy_yz_ptr,
                holonomy_zt_ptr,
                holonomy_ty_ptr,
                holonomy_xz_ptr,
                t_size,
                x_size,
                y_size,
                z_size,
                t_min,
                x_min,
                y_min,
                z_min,
                t_step,
                x_step,
                y_step,
                z_step
            ] __device__ (thrust::tuple<size_t, size_t, size_t, size_t, size_t> const &indices) -> void {
                auto const &[index, t_index, x_index, y_index, z_index] = indices;

                float const t = t_min + static_cast<float>(t_index) * t_step;
                float const x = x_min + static_cast<float>(x_index) * x_step;
                float const y = y_min + static_cast<float>(y_index) * y_step;
                float const z = z_min + static_cast<float>(z_index) * z_step;

                constexpr auto et = e(c<1>);
                constexpr auto ex = e(c<3>);
                constexpr auto ey = e(c<4>);
                constexpr auto ez = e(c<5>);

                auto const t_direction = t_step * et;
                auto const x_direction = x_step * ex;
                auto const y_direction = y_step * ey;
                auto const z_direction = z_step * ez;

                auto const site_position_0 = t * et + x * ex + y * ey + z * ez;
                auto const site_position_t = site_position_0 + t_direction;
                auto const site_position_x = site_position_0 + x_direction;
                auto const site_position_y = site_position_0 + y_direction;
                auto const site_position_z = site_position_0 + z_direction;

                auto const site_gauge_t_0 = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_0)));
                auto const site_gauge_x_0 = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_0)));
                auto const site_gauge_y_0 = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_0)));
                auto const site_gauge_z_0 = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_0)));
                auto const site_gauge_t_x = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_x)));
                auto const site_gauge_x_t = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_t)));
                auto const site_gauge_x_y = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_y)));
                auto const site_gauge_y_x = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_x)));
                auto const site_gauge_y_z = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_z)));
                auto const site_gauge_z_y = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_y)));
                auto const site_gauge_z_t = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_t)));
                auto const site_gauge_t_z = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_z)));
                auto const site_gauge_t_y = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_y)));
                auto const site_gauge_y_t = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_t)));
                auto const site_gauge_x_z = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_z)));
                auto const site_gauge_z_x = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_x)));

                holonomy_tx_ptr[index] = site_gauge_t_0 * site_gauge_x_t * inv(site_gauge_t_x) * inv(site_gauge_x_0);
                holonomy_xy_ptr[index] = site_gauge_x_0 * site_gauge_y_x * inv(site_gauge_x_y) * inv(site_gauge_y_0);
                holonomy_yz_ptr[index] = site_gauge_y_0 * site_gauge_z_y * inv(site_gauge_y_z) * inv(site_gauge_z_0);
                holonomy_zt_ptr[index] = site_gauge_z_0 * site_gauge_t_z * inv(site_gauge_z_t) * inv(site_gauge_t_0);
                holonomy_ty_ptr[index] = site_gauge_t_0 * site_gauge_y_t * inv(site_gauge_t_y) * inv(site_gauge_y_0);
                holonomy_xz_ptr[index] = site_gauge_x_0 * site_gauge_z_x * inv(site_gauge_x_z) * inv(site_gauge_z_0);
            }
        );

        m_info = ComputationInfo::Success;

        return *this;
    }
};