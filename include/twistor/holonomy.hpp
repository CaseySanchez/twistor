/**
 * @file holonomy.hpp
 * @brief Compute the holonomy on a lattice.
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include "twistor/csta.hpp"
#include "twistor/computation_info.hpp"

#include "lattice.hpp"

#ifdef TWISTOR_HDF5
#include <H5Cpp.h>
#endif

namespace Twistor::Device
{
    class Holonomy;
}

namespace Twistor::Host
{
    class Holonomy;
}

#ifdef TWISTOR_HDF5
namespace Twistor::HDF5
{
    void read(H5::Group const &group, Twistor::Host::Holonomy &holonomy);
    void write(H5::Group &group, Twistor::Host::Holonomy const &holonomy);
}
#endif

namespace Twistor::Device
{
    template <typename GaugeFieldFunctor>
    concept GaugeFieldFunctorConcept = requires(GaugeFieldFunctor gauge_field_functor, vector_t vector)
    {
        { gauge_field_functor(vector) } -> std::same_as<vector_t>;
    };

    class Holonomy
    {
        Lattice m_lattice;

        thrust::device_vector<motor_t> m_holonomy_tx;
        thrust::device_vector<motor_t> m_holonomy_xy;
        thrust::device_vector<motor_t> m_holonomy_yz;
        thrust::device_vector<motor_t> m_holonomy_zt;
        thrust::device_vector<motor_t> m_holonomy_ty;
        thrust::device_vector<motor_t> m_holonomy_xz;

        ComputationInfo m_info;

    public:
        /**
         * @brief Default constructor.
         */
        Holonomy() = default;

        /**
         * @brief Constructs a device holonomy object.
         * @param [in] lattice Spacetime lattice over which to compute the holonomy.
         */
        Holonomy(Lattice const &lattice);

        /**
         * @brief Constructs a device holonomy object from a host holonomy object.
         */
        Holonomy(Twistor::Host::Holonomy const &holonomy);

        /**
         * @brief Returns the status of the most recent call to `Holonomy::compute`.
         */
        ComputationInfo info() const;

        /**
         * @brief Returns the lattice.
         */
        Lattice lattice() const;

        /**
         * @brief Device buffer of tx-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_tx() const;

        /**
         * @brief Device buffer of xy-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_xy() const;

        /**
         * @brief Device buffer of yz-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_yz() const;

        /**
         * @brief Device buffer of zt-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_zt() const;

        /**
         * @brief Device buffer of ty-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_ty() const;

        /**
         * @brief Device buffer of xz-plane plaquette holonomies.
         */
        thrust::device_vector<motor_t> const &holonomy_xz() const;

        /**
         * @brief Schedules the holonomy kernel on the CUDA stream.
         * @param [in] gauge_field_functor Device-callable functor satisfying `GaugeFieldFunctorConcept`.
         * @return Reference to `this`.
         */
        template <GaugeFieldFunctorConcept GaugeFieldFunctor>
        Holonomy &compute(GaugeFieldFunctor gauge_field_functor);
    };
}

namespace Twistor::Host
{
    class Holonomy
    {
        Lattice m_lattice;

        thrust::host_vector<motor_t> m_holonomy_tx;
        thrust::host_vector<motor_t> m_holonomy_xy;
        thrust::host_vector<motor_t> m_holonomy_yz;
        thrust::host_vector<motor_t> m_holonomy_zt;
        thrust::host_vector<motor_t> m_holonomy_ty;
        thrust::host_vector<motor_t> m_holonomy_xz;

    public:
        /**
         * @brief Default constructor.
         */
        Holonomy() = default;

        /**
         * @brief Constructs a host holonomy object from a device holonomy object.
         */
        Holonomy(Twistor::Device::Holonomy const &holonomy);

        /**
         * @brief Returns the lattice.
         */
        Lattice lattice() const;

        /**
         * @brief Host buffer of tx-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_tx() const;

        /**
         * @brief Host buffer of xy-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_xy() const;

        /**
         * @brief Host buffer of yz-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_yz() const;

        /**
         * @brief Host buffer of zt-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_zt() const;

        /**
         * @brief Host buffer of ty-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_ty() const;

        /**
         * @brief Host buffer of xz-plane plaquette holonomies.
         */
        thrust::host_vector<motor_t> const &holonomy_xz() const;

#ifdef TWISTOR_HDF5
    private:
        friend void Twistor::HDF5::read(H5::Group const &group, Twistor::Host::Holonomy &holonomy);
        friend void Twistor::HDF5::write(H5::Group &group, Twistor::Host::Holonomy const &holonomy);
#endif
    };
}

inline Twistor::Device::Holonomy::Holonomy(Lattice const &lattice) :
    m_lattice { lattice },
    m_info { ComputationInfo::InvalidInput }
{
}

inline Twistor::Device::Holonomy::Holonomy(Twistor::Host::Holonomy const &holonomy)
{
    m_lattice = holonomy.lattice();

    m_holonomy_tx = holonomy.holonomy_tx();
    m_holonomy_xy = holonomy.holonomy_xy();
    m_holonomy_yz = holonomy.holonomy_yz();
    m_holonomy_zt = holonomy.holonomy_zt();
    m_holonomy_ty = holonomy.holonomy_ty();
    m_holonomy_xz = holonomy.holonomy_xz();
}

inline Twistor::ComputationInfo Twistor::Device::Holonomy::info() const
{
    return m_info;
}

inline Twistor::Lattice Twistor::Device::Holonomy::lattice() const
{
    return m_lattice;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_tx() const
{
    return m_holonomy_tx;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_xy() const
{
    return m_holonomy_xy;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_yz() const
{
    return m_holonomy_yz;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_zt() const
{
    return m_holonomy_zt;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_ty() const
{
    return m_holonomy_ty;
}

inline thrust::device_vector<motor_t> const &Twistor::Device::Holonomy::holonomy_xz() const
{
    return m_holonomy_xz;
}

template <Twistor::Device::GaugeFieldFunctorConcept GaugeFieldFunctor>
inline Twistor::Device::Holonomy &Twistor::Device::Holonomy::compute(GaugeFieldFunctor gauge_field_functor)
{
    size_t const t_size = m_lattice.t_size();
    size_t const x_size = m_lattice.x_size();
    size_t const y_size = m_lattice.y_size();
    size_t const z_size = m_lattice.z_size();

    float const t_min = m_lattice.t_min();
    float const x_min = m_lattice.x_min();
    float const y_min = m_lattice.y_min();
    float const z_min = m_lattice.z_min();

    float const t_max = m_lattice.t_max();
    float const x_max = m_lattice.x_max();
    float const y_max = m_lattice.y_max();
    float const z_max = m_lattice.z_max();

    float const t_step = m_lattice.t_step();
    float const x_step = m_lattice.x_step();
    float const y_step = m_lattice.y_step();
    float const z_step = m_lattice.z_step();

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
        thrust::device,
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
            constexpr auto et = e(c<1>);
            constexpr auto ex = e(c<3>);
            constexpr auto ey = e(c<4>);
            constexpr auto ez = e(c<5>);

            auto const &[index, t_index, x_index, y_index, z_index] = indices;

            float const t = t_min + static_cast<float>(t_index) * t_step;
            float const x = x_min + static_cast<float>(x_index) * x_step;
            float const y = y_min + static_cast<float>(y_index) * y_step;
            float const z = z_min + static_cast<float>(z_index) * z_step;

            auto const t_direction = t_step * et;
            auto const x_direction = x_step * ex;
            auto const y_direction = y_step * ey;
            auto const z_direction = z_step * ez;

            auto const site_position_0 = t * et + x * ex + y * ey + z * ez;
            auto const site_position_t = site_position_0 + t_direction;
            auto const site_position_x = site_position_0 + x_direction;
            auto const site_position_y = site_position_0 + y_direction;
            auto const site_position_z = site_position_0 + z_direction;

            auto const site_motor_t_0 = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_0)));
            auto const site_motor_x_0 = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_0)));
            auto const site_motor_y_0 = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_0)));
            auto const site_motor_z_0 = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_0)));
            auto const site_motor_t_x = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_x)));
            auto const site_motor_x_t = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_t)));
            auto const site_motor_x_y = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_y)));
            auto const site_motor_y_x = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_x)));
            auto const site_motor_y_z = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_z)));
            auto const site_motor_z_y = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_y)));
            auto const site_motor_z_t = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_t)));
            auto const site_motor_t_z = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_z)));
            auto const site_motor_t_y = exp(0.5 * (t_direction ^ gauge_field_functor(site_position_y)));
            auto const site_motor_y_t = exp(0.5 * (y_direction ^ gauge_field_functor(site_position_t)));
            auto const site_motor_x_z = exp(0.5 * (x_direction ^ gauge_field_functor(site_position_z)));
            auto const site_motor_z_x = exp(0.5 * (z_direction ^ gauge_field_functor(site_position_x)));

            holonomy_tx_ptr[index] = unit(site_motor_t_0 * site_motor_x_t * inv(site_motor_t_x) * inv(site_motor_x_0));
            holonomy_xy_ptr[index] = unit(site_motor_x_0 * site_motor_y_x * inv(site_motor_x_y) * inv(site_motor_y_0));
            holonomy_yz_ptr[index] = unit(site_motor_y_0 * site_motor_z_y * inv(site_motor_y_z) * inv(site_motor_z_0));
            holonomy_zt_ptr[index] = unit(site_motor_z_0 * site_motor_t_z * inv(site_motor_z_t) * inv(site_motor_t_0));
            holonomy_ty_ptr[index] = unit(site_motor_t_0 * site_motor_y_t * inv(site_motor_t_y) * inv(site_motor_y_0));
            holonomy_xz_ptr[index] = unit(site_motor_x_0 * site_motor_z_x * inv(site_motor_x_z) * inv(site_motor_z_0));
        }
    );

    m_info = ComputationInfo::Success;

    return *this;
}

inline Twistor::Host::Holonomy::Holonomy(Twistor::Device::Holonomy const &holonomy)
{
    m_lattice = holonomy.lattice();

    m_holonomy_tx = holonomy.holonomy_tx();
    m_holonomy_xy = holonomy.holonomy_xy();
    m_holonomy_yz = holonomy.holonomy_yz();
    m_holonomy_zt = holonomy.holonomy_zt();
    m_holonomy_ty = holonomy.holonomy_ty();
    m_holonomy_xz = holonomy.holonomy_xz();
}

inline Twistor::Lattice Twistor::Host::Holonomy::lattice() const
{
    return m_lattice;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_tx() const
{
    return m_holonomy_tx;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_xy() const
{
    return m_holonomy_xy;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_yz() const
{
    return m_holonomy_yz;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_zt() const
{
    return m_holonomy_zt;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_ty() const
{
    return m_holonomy_ty;
}

inline thrust::host_vector<motor_t> const &Twistor::Host::Holonomy::holonomy_xz() const
{
    return m_holonomy_xz;
}

#ifdef TWISTOR_HDF5
namespace Twistor::HDF5
{
    void read(H5::Group const &group, Twistor::Host::Holonomy &holonomy)
    {
        std::array<hsize_t, 2> holonomy_tx_dims;
        std::array<hsize_t, 2> holonomy_xy_dims;
        std::array<hsize_t, 2> holonomy_yz_dims;
        std::array<hsize_t, 2> holonomy_zt_dims;
        std::array<hsize_t, 2> holonomy_ty_dims;
        std::array<hsize_t, 2> holonomy_xz_dims;

        group.openDataSet("holonomy_tx").getSpace().getSimpleExtentDims(holonomy_tx_dims.data());
        group.openDataSet("holonomy_xy").getSpace().getSimpleExtentDims(holonomy_xy_dims.data());
        group.openDataSet("holonomy_yz").getSpace().getSimpleExtentDims(holonomy_yz_dims.data());
        group.openDataSet("holonomy_zt").getSpace().getSimpleExtentDims(holonomy_zt_dims.data());
        group.openDataSet("holonomy_ty").getSpace().getSimpleExtentDims(holonomy_ty_dims.data());
        group.openDataSet("holonomy_xz").getSpace().getSimpleExtentDims(holonomy_xz_dims.data());

        holonomy.m_holonomy_tx.resize(holonomy_tx_dims[0]);
        holonomy.m_holonomy_xy.resize(holonomy_xy_dims[0]);
        holonomy.m_holonomy_yz.resize(holonomy_yz_dims[0]);
        holonomy.m_holonomy_zt.resize(holonomy_zt_dims[0]);
        holonomy.m_holonomy_ty.resize(holonomy_ty_dims[0]);
        holonomy.m_holonomy_xz.resize(holonomy_xz_dims[0]);

        group.openDataSet("holonomy_tx").read(holonomy.m_holonomy_tx.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("holonomy_xy").read(holonomy.m_holonomy_xy.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("holonomy_yz").read(holonomy.m_holonomy_yz.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("holonomy_zt").read(holonomy.m_holonomy_zt.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("holonomy_ty").read(holonomy.m_holonomy_ty.data(), H5::PredType::NATIVE_FLOAT);
        group.openDataSet("holonomy_xz").read(holonomy.m_holonomy_xz.data(), H5::PredType::NATIVE_FLOAT);
    }

    void write(H5::Group &group, Twistor::Host::Holonomy const &holonomy)
    {
        std::array<hsize_t, 2> const holonomy_tx_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_tx.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        std::array<hsize_t, 2> const holonomy_xy_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_xy.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        std::array<hsize_t, 2> const holonomy_yz_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_yz.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        std::array<hsize_t, 2> const holonomy_zt_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_zt.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        std::array<hsize_t, 2> const holonomy_ty_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_ty.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        std::array<hsize_t, 2> const holonomy_xz_dims = {
            static_cast<hsize_t>(holonomy.m_holonomy_xz.size()),
            static_cast<hsize_t>(sizeof(motor_t) / sizeof(float))
        };

        H5::DataSpace const holonomy_tx_dataspace(holonomy_tx_dims.size(), holonomy_tx_dims.data());
        H5::DataSpace const holonomy_xy_dataspace(holonomy_xy_dims.size(), holonomy_xy_dims.data());
        H5::DataSpace const holonomy_yz_dataspace(holonomy_yz_dims.size(), holonomy_yz_dims.data());
        H5::DataSpace const holonomy_zt_dataspace(holonomy_zt_dims.size(), holonomy_zt_dims.data());
        H5::DataSpace const holonomy_ty_dataspace(holonomy_ty_dims.size(), holonomy_ty_dims.data());
        H5::DataSpace const holonomy_xz_dataspace(holonomy_xz_dims.size(), holonomy_xz_dims.data());

        group.createDataSet("holonomy_tx", H5::PredType::NATIVE_FLOAT, holonomy_tx_dataspace).write(holonomy.m_holonomy_tx.data(), H5::PredType::NATIVE_FLOAT);
        group.createDataSet("holonomy_xy", H5::PredType::NATIVE_FLOAT, holonomy_xy_dataspace).write(holonomy.m_holonomy_xy.data(), H5::PredType::NATIVE_FLOAT);
        group.createDataSet("holonomy_yz", H5::PredType::NATIVE_FLOAT, holonomy_yz_dataspace).write(holonomy.m_holonomy_yz.data(), H5::PredType::NATIVE_FLOAT);
        group.createDataSet("holonomy_zt", H5::PredType::NATIVE_FLOAT, holonomy_zt_dataspace).write(holonomy.m_holonomy_zt.data(), H5::PredType::NATIVE_FLOAT);
        group.createDataSet("holonomy_ty", H5::PredType::NATIVE_FLOAT, holonomy_ty_dataspace).write(holonomy.m_holonomy_ty.data(), H5::PredType::NATIVE_FLOAT);
        group.createDataSet("holonomy_xz", H5::PredType::NATIVE_FLOAT, holonomy_xz_dataspace).write(holonomy.m_holonomy_xz.data(), H5::PredType::NATIVE_FLOAT);
    }
}
#endif