// SPDX-FileCopyrightText: 2026 Baptiste Legouix
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <ddc/ddc.hpp>

#include <similie/tensor/identity_tensor.hpp>

#include "exterior.hpp"

template <class... CDim>
using MetricIndex = sil::tensor::TensorIdentityIndex<
        sil::tensor::Covariant<sil::tensor::MetricIndex1<CDim...>>,
        sil::tensor::Covariant<sil::tensor::MetricIndex2<CDim...>>>;

struct X
{
};

struct Y
{
};

struct DDimX : ddc::UniformPointSampling<X>
{
};

struct DDimY : ddc::UniformPointSampling<Y>
{
};

struct Mu2 : sil::tensor::TensorNaturalIndex<X, Y>
{
};

using ScalarIndex = sil::tensor::Covariant<sil::tensor::ScalarIndex>;
using PositionIndex = sil::tensor::Contravariant<sil::tensor::TensorNaturalIndex<X, Y>>;
using MetricTensorIndex = MetricIndex<X, Y>;

namespace {

template <class TensorType>
auto compute_full_domain_gradient_matrix_storage(TensorType potential)
{
    using DerivativeIndex = Mu2;
    [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeIndex> derivative_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, DerivativeIndex>
            derivative_dom(potential.non_indices_domain(), derivative_accessor.domain());
    ddc::Chunk derivative_alloc(derivative_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor derivative(derivative_alloc);
    return std::make_pair(std::move(derivative_alloc), derivative);
}

template <class TensorType, class MetricType, class PositionType>
void compute_full_domain_laplacian(
        TensorType laplacian,
        TensorType potential,
        MetricType metric,
        PositionType position)
{
    using LaplacianDummyIndex2
            = sil::tensor::Covariant<sil::exterior::detail::LaplacianDummy2<Mu2>>;
    using DerivativeIndex = sil::exterior::coboundary_index_t<LaplacianDummyIndex2, ScalarIndex>;
    using DerivativeMuUpSeq = sil::tensor::upper_t<
            ddc::to_type_seq_t<sil::tensor::natural_domain_t<DerivativeIndex>>>;
    using DerivativeNuLowSeq = typename sil::exterior::detail::CodifferentialDummyIndexSeq<
            LaplacianDummyIndex2::size() - DerivativeIndex::rank(),
            LaplacianDummyIndex2>::type;
    using DerivativeRhoLowSeq
            = ddc::type_seq_merge_t<ddc::detail::TypeSeq<LaplacianDummyIndex2>, DerivativeNuLowSeq>;
    using DerivativeRhoUpSeq = sil::tensor::upper_t<DerivativeRhoLowSeq>;
    using DerivativeSigmaLowSeq = ddc::type_seq_remove_t<
            sil::tensor::lower_t<DerivativeMuUpSeq>,
            ddc::detail::TypeSeq<LaplacianDummyIndex2>>;
    using DerivativeDualIndex = sil::misc::
            convert_type_seq_to_t<sil::tensor::TensorAntisymmetricIndex, DerivativeNuLowSeq>;

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
            derivative_hodge_star_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            sil::exterior::hodge_star_domain_t<DerivativeMuUpSeq, DerivativeNuLowSeq>>
            derivative_hodge_star_dom(
                    metric.non_indices_domain(),
                    derivative_hodge_star_accessor.domain());
    ddc::Chunk derivative_hodge_star_alloc(derivative_hodge_star_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor derivative_hodge_star(derivative_hodge_star_alloc);

    [[maybe_unused]] sil::tensor::tensor_accessor_for_domain_t<
            sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
            dual_derivative_hodge_star_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            sil::exterior::hodge_star_domain_t<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>>
            dual_derivative_hodge_star_dom(
                    metric.non_indices_domain(),
                    dual_derivative_hodge_star_accessor.domain());
    ddc::Chunk dual_derivative_hodge_star_alloc(
            dual_derivative_hodge_star_dom,
            ddc::HostAllocator<double>());
    sil::tensor::Tensor dual_derivative_hodge_star(dual_derivative_hodge_star_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<DerivativeDualIndex>
            derivative_dual_tensor_accessor;
    ddc::cartesian_prod_t<
            typename TensorType::non_indices_domain_t,
            ddc::DiscreteDomain<DerivativeDualIndex>>
            derivative_dual_tensor_dom(
                    potential.non_indices_domain(),
                    derivative_dual_tensor_accessor.domain());
    ddc::Chunk
            derivative_dual_tensor_alloc(derivative_dual_tensor_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor derivative_dual_tensor_buffer(derivative_dual_tensor_alloc);

    sil::exterior::fill_discrete_hodge_star<DerivativeMuUpSeq, DerivativeNuLowSeq>(
            Kokkos::DefaultHostExecutionSpace(),
            derivative_hodge_star,
            metric,
            position);
    sil::exterior::fill_discrete_hodge_star<DerivativeRhoUpSeq, DerivativeSigmaLowSeq>(
            Kokkos::DefaultHostExecutionSpace(),
            dual_derivative_hodge_star,
            metric,
            position);

    sil::exterior::laplacian<MetricTensorIndex, Mu2, ScalarIndex>(
            Kokkos::DefaultHostExecutionSpace(),
            laplacian,
            potential,
            derivative_hodge_star,
            dual_derivative_hodge_star,
            derivative_dual_tensor_buffer);
}

std::vector<double> jacobi_eigenvalues(std::vector<double> matrix, int n)
{
    auto idx = [n](int i, int j) { return i * n + j; };
    for (int iter = 0; iter < 100 * n * n; ++iter) {
        int p = 0;
        int q = 1;
        double max_abs = 0.0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double const val = std::abs(matrix[idx(i, j)]);
                if (val > max_abs) {
                    max_abs = val;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_abs < 1e-12) {
            break;
        }

        double const app = matrix[idx(p, p)];
        double const aqq = matrix[idx(q, q)];
        double const apq = matrix[idx(p, q)];
        double const tau = (aqq - app) / (2.0 * apq);
        double const t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        double const c = 1.0 / std::sqrt(1.0 + t * t);
        double const s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == p || k == q) {
                continue;
            }
            double const aik = matrix[idx(k, p)];
            double const akq = matrix[idx(k, q)];
            matrix[idx(k, p)] = c * aik - s * akq;
            matrix[idx(p, k)] = matrix[idx(k, p)];
            matrix[idx(k, q)] = s * aik + c * akq;
            matrix[idx(q, k)] = matrix[idx(k, q)];
        }

        matrix[idx(p, p)] = app - t * apq;
        matrix[idx(q, q)] = aqq + t * apq;
        matrix[idx(p, q)] = 0.0;
        matrix[idx(q, p)] = 0.0;
    }

    std::vector<double> eigs(n);
    for (int i = 0; i < n; ++i) {
        eigs[i] = matrix[idx(i, i)];
    }
    std::sort(eigs.begin(), eigs.end());
    return eigs;
}

} // namespace

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard const kokkos_scope(argc, argv);
    ddc::ScopeGuard const ddc_scope(argc, argv);

    ddc::Coordinate<X, Y> lower_bounds(-1.0, -1.0);
    ddc::Coordinate<X, Y> upper_bounds(1.0, 1.0);
    ddc::DiscreteVector<DDimX, DDimY> nb_cells(4, 4);
    ddc::DiscreteDomain<DDimX> mesh_x = ddc::init_discrete_space<DDimX>(DDimX::init<DDimX>(
            ddc::Coordinate<X>(lower_bounds),
            ddc::Coordinate<X>(upper_bounds),
            ddc::DiscreteVector<DDimX>(nb_cells)));
    ddc::DiscreteDomain<DDimY> mesh_y = ddc::init_discrete_space<DDimY>(DDimY::init<DDimY>(
            ddc::Coordinate<Y>(lower_bounds),
            ddc::Coordinate<Y>(upper_bounds),
            ddc::DiscreteVector<DDimY>(nb_cells)));
    ddc::DiscreteDomain<DDimX, DDimY> mesh_xy(mesh_x, mesh_y);

    [[maybe_unused]] sil::tensor::TensorAccessor<ScalarIndex> scalar_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, ScalarIndex> scalar_dom(mesh_xy, scalar_accessor.domain());
    ddc::Chunk potential_alloc(scalar_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor potential(potential_alloc);
    ddc::Chunk laplacian_alloc(scalar_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor laplacian(laplacian_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<MetricTensorIndex> metric_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, MetricTensorIndex>
            metric_dom(mesh_xy, metric_accessor.domain());
    ddc::Chunk metric_alloc(metric_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor metric(metric_alloc);

    [[maybe_unused]] sil::tensor::TensorAccessor<PositionIndex> position_accessor;
    ddc::DiscreteDomain<DDimX, DDimY, PositionIndex>
            position_dom(mesh_xy, position_accessor.domain());
    ddc::Chunk position_alloc(position_dom, ddc::HostAllocator<double>());
    sil::tensor::Tensor position(position_alloc);

    std::vector<ddc::DiscreteElement<DDimX, DDimY>> elems;
    ddc::host_for_each(mesh_xy, [&](ddc::DiscreteElement<DDimX, DDimY> elem) {
        elems.push_back(elem);
        position(elem, position_accessor.template access_element<X>())
                = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimX>(elem)));
        position(elem, position_accessor.template access_element<Y>())
                = static_cast<double>(ddc::coordinate(ddc::DiscreteElement<DDimY>(elem)));
    });

    int const n = static_cast<int>(elems.size());
    std::vector<double> a(n * n, 0.0);
    std::vector<double> dtd(n * n, 0.0);
    std::vector<double> gradmat(2 * n * n, 0.0);
    std::vector<double> mass0(n, 0.0);
    std::array<std::size_t, 0> empty_ids {};
    auto [gradient_alloc, gradient] = compute_full_domain_gradient_matrix_storage(potential);

    for (int i = 0; i < n; ++i) {
        mass0[i] = sil::exterior::DualSimplexVolume<
                sil::exterior::DualStrategy::Circumcentric,
                2,
                decltype(metric),
                decltype(position),
                ddc::DiscreteElement<DDimX, DDimY>>::run(metric, position, elems[i], empty_ids);
    }

    for (int col = 0; col < n; ++col) {
        ddc::host_for_each(
                potential.domain(),
                [&](ddc::DiscreteElement<DDimX, DDimY, ScalarIndex> elem) {
                    potential(elem) = 0.0;
                });
        potential(elems[col], ddc::DiscreteElement<ScalarIndex>(0)) = 1.0;
        sil::exterior::
                deriv<Mu2, ScalarIndex>(Kokkos::DefaultHostExecutionSpace(), gradient, potential);
        compute_full_domain_laplacian(laplacian, potential, metric, position);
        for (int row = 0; row < n; ++row) {
            a[row * n + col] = laplacian(elems[row], ddc::DiscreteElement<ScalarIndex>(0));
        }
        for (int edge = 0; edge < n; ++edge) {
            gradmat[((2 * edge + 0) * n) + col]
                    = gradient(elems[edge], gradient.accessor().template access_element<X>());
            gradmat[((2 * edge + 1) * n) + col]
                    = gradient(elems[edge], gradient.accessor().template access_element<Y>());
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double value = 0.0;
            for (int edge_dof = 0; edge_dof < 2 * n; ++edge_dof) {
                value += gradmat[edge_dof * n + i] * gradmat[edge_dof * n + j];
            }
            dtd[i * n + j] = value;
        }
    }

    std::vector<double> b(n * n, 0.0);
    double max_constant_residual = 0.0;
    double max_weighted_asymmetry = 0.0;
    int asym_i = 0;
    int asym_j = 0;
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            b[i * n + j] = mass0[i] * a[i * n + j];
            row_sum += a[i * n + j];
        }
        max_constant_residual = std::max(max_constant_residual, std::abs(row_sum));
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double const asym = std::abs(b[i * n + j] - b[j * n + i]);
            if (asym > max_weighted_asymmetry) {
                max_weighted_asymmetry = asym;
                asym_i = i;
                asym_j = j;
            }
        }
    }

    std::vector<double> sym_b = b;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double const avg = 0.5 * (b[i * n + j] + b[j * n + i]);
            sym_b[i * n + j] = avg;
            sym_b[j * n + i] = avg;
        }
    }
    std::vector<double> eigs = jacobi_eigenvalues(sym_b, n);

    std::cout << std::setprecision(16);
    std::cout << "Mass0 diagonal:\n";
    for (int i = 0; i < n; ++i) {
        std::cout << i << " (" << elems[i].uid<DDimX>() << "," << elems[i].uid<DDimY>() << ") "
                  << mass0[i] << "\n";
    }
    std::cout << "Max constant residual of delta d on ones: " << max_constant_residual << "\n";
    std::cout << "Max weighted asymmetry of M0*(delta d): " << max_weighted_asymmetry << "\n";
    std::cout << "Worst asymmetry pair: (" << asym_i << "," << asym_j << ") => "
              << b[asym_i * n + asym_j] << " vs " << b[asym_j * n + asym_i] << "\n";
    std::cout << "Smallest eigenvalue of sym(M0*(delta d)): " << eigs.front() << "\n";
    std::cout << "Largest eigenvalue of sym(M0*(delta d)): " << eigs.back() << "\n";
    std::cout << "Matrix A = delta d:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << a[i * n + j] << (j + 1 == n ? '\n' : ' ');
        }
    }
    std::cout << "Matrix D^T D:\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << dtd[i * n + j] << (j + 1 == n ? '\n' : ' ');
        }
    }

    ddc::detail::g_discrete_space_dual<DDimX>.reset();
    ddc::detail::g_discrete_space_dual<DDimY>.reset();
    return 0;
}
