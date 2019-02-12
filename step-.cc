/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2017 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Thomas Clevenger, Clemson University
 *         Timo Heister, University of Utah
 */

#include <deal.II/base/work_stream.h>
#include <deal.II/meshworker/mesh_loop.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/path_search.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/relaxation_block.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>
#include<deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>



#include <boost/random.hpp>
#include <boost/random/uniform_int_distribution.hpp>



namespace Step100
{
using namespace dealii;


template <int dim>
struct ScratchData
{
    ScratchData(const FiniteElement<dim> &fe,
                const unsigned int        quadrature_degree)
        : fe_values(fe,
                    QGauss<dim>(quadrature_degree),
                    update_values | update_gradients | update_hessians | update_quadrature_points |
                    update_JxW_values)
    {}

    ScratchData(const ScratchData<dim> &scratch_data)
        : fe_values(scratch_data.fe_values.get_fe(),
                    scratch_data.fe_values.get_quadrature(),
                    update_values | update_gradients | update_hessians | update_quadrature_points |
                    update_JxW_values)
    {}

    FEValues<dim> fe_values;
};
struct CopyData
{
    unsigned int level;
    unsigned int dofs_per_cell;

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
};


struct Settings
{
    bool try_parse(const std::string &prm_filename);

    unsigned int fe_degree;
    std::string smoother_type;
    std::string dof_renum;
    bool with_sd;
    bool output;
};
bool
Settings::try_parse(const std::string &prm_filename)
{
    ParameterHandler prm;

    prm.declare_entry("fe degree", "1",
                      Patterns::Integer(0),
                      "Finite Element degree");
    prm.declare_entry("smoother type", "block sor",
                      Patterns::Selection("sor|jacobi|block sor|block jacobi"),
                      "Smoother Type: sor|jacobi|block sor|block jacobi");
    prm.declare_entry("dof renumbering", "downstream",
                      Patterns::Selection("no|random|downstream|upstream"),
                      "Dof renumbering: no|random|downstream|upstream");
    prm.declare_entry("with sd", "true",
                      Patterns::Bool(),
                      "With streamline diffusion: true|false");
    prm.declare_entry("output", "true",
                      Patterns::Bool(),
                      "Output: true|false");
    try
    {
        prm.parse_input(prm_filename);
    }
    catch (const dealii::PathSearch::ExcFileNotFound &)
    {
      if (prm_filename.size()>0)
	std::cerr << "ERRROR: could not open the .prm file '"
		  << prm_filename << "'" << std::endl;
      else
	std::cerr << "Usage: please pass a .prm file as the first argument"
		  << std::endl;

      prm.print_parameters(std::cout, ParameterHandler::Text);
        return false;
    }
    this->fe_degree = prm.get_integer("fe degree");
    this->smoother_type = prm.get("smoother type");
    this->dof_renum = prm.get("dof renumbering");
    this->with_sd = prm.get_bool("with sd");
    this->output = prm.get_bool("output");

    return true;
}


template <class Iterator, int dim>
struct CompareDownstream
{
    /**
   * Constructor.
   */
    CompareDownstream (const Tensor<1,dim> &dir)
        :
          dir(dir)
    {}
    /**
   * Return true if c1 less c2.
   */
    bool operator () (const Iterator &c1, const Iterator &c2) const
    {
        const Tensor<1,dim> diff = c2->center() - c1->center();
        return (diff*dir > 0);
    }

private:
    /**
   * Flow direction.
   */
    const Tensor<1,dim> dir;
};


namespace
{
template <int dim>
std::vector<unsigned int>
create_downstream_order (const DoFHandler<dim> &dof,
                         const Tensor<1,dim> direction,
                         const unsigned int     level)
{
    std::vector<typename DoFHandler<dim>::level_cell_iterator> ordered_cells;
    ordered_cells.reserve (dof.get_triangulation().n_cells(level));
    const CompareDownstream<typename DoFHandler<dim>::level_cell_iterator,dim> comparator(direction);

    typename DoFHandler<dim>::level_cell_iterator cell = dof.begin(level);
    typename DoFHandler<dim>::level_cell_iterator endc = dof.end(level);
    for (; cell!=endc; ++cell)
        ordered_cells.push_back(cell);

    std::sort (ordered_cells.begin(), ordered_cells.end(), comparator);

    std::vector<unsigned > ordered_indices;
    ordered_indices.reserve (dof.get_triangulation().n_cells(level));

    for (unsigned int i=0; i<ordered_cells.size(); ++i)
        ordered_indices.push_back(ordered_cells[i]->index());

    return ordered_indices;
}

template <int dim>
std::vector<unsigned int>
create_downstream_order (const DoFHandler<dim> &dof,
                         const Tensor<1,dim> direction)
{
    std::vector<typename DoFHandler<dim>::active_cell_iterator> ordered_cells;
    ordered_cells.reserve (dof.get_triangulation().n_active_cells());
    const CompareDownstream<typename DoFHandler<dim>::active_cell_iterator,dim> comparator(direction);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active();
    typename DoFHandler<dim>::active_cell_iterator endc = dof.end();
    for (; cell!=endc; ++cell)
        ordered_cells.push_back(cell);

    std::sort (ordered_cells.begin(), ordered_cells.end(), comparator);

    std::vector<unsigned int> ordered_indices;
    ordered_indices.reserve (dof.get_triangulation().n_active_cells());

    for (unsigned int i=0; i<ordered_cells.size(); ++i)
        ordered_indices.push_back(ordered_cells[i]->index());

    return ordered_indices;
}



template <int dim>
std::vector<unsigned int>
create_random_order (const DoFHandler<dim> &dof,
                     const unsigned int level)
{
    const unsigned int n_cells = dof.get_triangulation().n_cells(level);

    std::vector<unsigned int> ordered_cells;
    ordered_cells.reserve (n_cells);

    typename DoFHandler<dim>::cell_iterator cell = dof.begin(level);
    typename DoFHandler<dim>::cell_iterator endc = dof.end(level);
    for (; cell!=endc; ++cell)
        ordered_cells.push_back(cell->index());

    // shuffle the elements; the following is essentially std::shuffle (which
    // is new in C++11) but with a boost URNG
    ::boost::mt19937 random_number_generator;
    for (unsigned int i = 1; i < n_cells; ++i)
    {
        // get a random number between 0 and i (inclusive)
        const unsigned int j =
                ::boost::random::uniform_int_distribution<>(0, i)(
                    random_number_generator);

        // if possible, swap the elements
        if (i != j)
            std::swap(ordered_cells[i], ordered_cells[j]);
    }

    return ordered_cells;
}

template <int dim>
std::vector<unsigned int>
create_random_order (const DoFHandler<dim> &dof)
{
    const unsigned int n_cells = dof.get_triangulation().n_active_cells();

    std::vector<unsigned int> ordered_cells;
    ordered_cells.reserve (n_cells);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active();
    typename DoFHandler<dim>::active_cell_iterator endc = dof.end();
    for (; cell!=endc; ++cell)
        ordered_cells.push_back(cell->index());

    // shuffle the elements; the following is essentially std::shuffle (which
    // is new in C++11) but with a boost URNG
    ::boost::mt19937 random_number_generator;
    for (unsigned int i = 1; i < n_cells; ++i)
    {
        // get a random number between 0 and i (inclusive)
        const unsigned int j =
                ::boost::random::uniform_int_distribution<>(0, i)(
                    random_number_generator);

        // if possible, swap the elements
        if (i != j)
            std::swap(ordered_cells[i], ordered_cells[j]);
    }

    return ordered_cells;
}

}


template <int dim>
class Epsilon : public Function<dim>
{
public:
    Epsilon () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;
};
template <int dim>
double Epsilon<dim>::value (const Point<dim> &p,
                            const unsigned int) const
{
    (void)p;
    return 0.005;
}
template <int dim>
void Epsilon<dim>::value_list (const std::vector<Point<dim> > &points,
                               std::vector<double>            &values,
                               const unsigned int              component) const
{
    (void)component;
    const unsigned int n_points = points.size();
    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));
    Assert (component == 0,
            ExcIndexRange (component, 0, 1));
    for (unsigned int i=0; i<n_points; ++i)
        values[i] = Epsilon<dim>::value (points[i]);
}




template <int dim>
class AdvectionField : public TensorFunction<1,dim>
{
public:
    AdvectionField () : TensorFunction<1,dim> () {}

    virtual Tensor<1,dim> value (const Point<dim> &p) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<1,dim> >    &values) const;

    DeclException2 (ExcDimensionMismatch,
                    unsigned int, unsigned int,
                    << "The vector has size " << arg1 << " but should have "
                    << arg2 << " elements.");
};
template <int dim>
Tensor<1,dim>
AdvectionField<dim>::value (const Point<dim> &p) const
{
    (void)p;
    Point<dim> value;

    value[0] = -1.0*std::sin(numbers::PI/6.0);
    if (dim > 1)
    {
        value[1] = std::cos(numbers::PI/6.0);
        if (dim > 2)
            value[2] = 1.0;
    }

    return value;
}
template <int dim>
void
AdvectionField<dim>::value_list (const std::vector<Point<dim> > &points,
                                 std::vector<Tensor<1,dim> >    &values) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = AdvectionField<dim>::value (points[i]);
}




template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;

private:
    static const Point<dim> center_point;
};

template <int dim>
double
RightHandSide<dim>::value (const Point<dim>   &p,
                           const unsigned int  component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    (void) component;
    (void) p;

    return 0.0;
}



template <int dim>
void
RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
                                std::vector<double>            &values,
                                const unsigned int              component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = RightHandSide<dim>::value (points[i], component);
}



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
    BoundaryValues () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;
};



template <int dim>
double
BoundaryValues<dim>::value (const Point<dim>   &p,
                            const unsigned int  component) const
{
    Assert (component == 0, ExcIndexRange (component, 0, 1));
    (void)component;

    if (std::fabs(p[0]*p[0] + p[1]*p[1] - 0.3*0.3) < 1e-8) //around cylinder
    {
        return 0.0;
    }
    else if (std::fabs(p[0]-1)<1e-8 // TODO: document
             ||
             (p[1]<0.999 && p[1]>=-std::sqrt(3)*(p[0]-0.5)-1))
    {
        return 1.0;
    }
    else
    {
        return 0.0;
    }
}



template <int dim>
void
BoundaryValues<dim>::value_list (const std::vector<Point<dim> > &points,
                                 std::vector<double>            &values,
                                 const unsigned int              component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));

    for (unsigned int i=0; i<points.size(); ++i)
        values[i] = BoundaryValues<dim>::value (points[i], component);
}

template <int dim>
double delta_value (const double hk,
                    const double eps,
                    const Tensor<1,dim> dir,
                    const double pk)
{
    double Peclet = dir.norm()*hk/(2.0*eps*pk);
    double coth = (1.0+std::exp(-2.0*Peclet))/(1.0-std::exp(-2.0*Peclet));

    return hk/(2.0*dir.norm()*pk)*(coth - 1.0/Peclet);
}


template <int dim>
class AdvectionProblem
{
public:
    AdvectionProblem (Settings settings);
    void run ();

private:
    void setup_system ();

    template <class IteratorType>
    void
    assemble_cell(const IteratorType &cell,
                  ScratchData<dim>   &scratch_data,
                  CopyData           &copy_data);
    void
    assemble_system_and_multigrid();

    std::shared_ptr<MGSmoother<Vector<double>>> create_smoother ();

    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    FE_Q<dim>            fe;
    MappingQ<dim> mapping;
    unsigned int quad_degree;

    AffineConstraints<double>     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

    MGLevelObject<SparsityPattern>       mg_sparsity_patterns;
    MGLevelObject<SparsityPattern>       mg_interface_sparsity_patterns;

    MGLevelObject<SparseMatrix<double> > mg_matrices;
    MGLevelObject<SparseMatrix<double> > mg_interface_in;
    MGLevelObject<SparseMatrix<double> > mg_interface_out;

    MGLevelObject<AffineConstraints<double>> mg_constraints;

    MGConstrainedDoFs                    mg_constrained_dofs;

    Settings settings;
};



template <int dim>
AdvectionProblem<dim>::AdvectionProblem (Settings settings)
    :
      triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
      dof_handler (triangulation),
      fe(settings.fe_degree),
      mapping(settings.fe_degree),
      quad_degree (2*fe.degree+2),
      settings(settings)
{}




template <int dim>
void AdvectionProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (fe);

    // Could renumber, but doesn't matter
//    if (settings.dof_renum == "random")
//    {
//        if (settings.smoother_type =="sor" ||
//                settings.smoother_type =="jacobi")
//            Assert(false,ExcMessage("Random renumbering for point-smoothers not yet implemented."));
//    }
//    else if (settings.dof_renum == "downstream")
//    {
//        const AdvectionField<dim> w;
//        DoFRenumbering::downstream(dof_handler,w.value(Point<dim>()));
//    }
//    else if (settings.dof_renum == "upstream")
//    {
//        const AdvectionField<dim> w;
//        DoFRenumbering::downstream(dof_handler,-1.0*w.value(Point<dim>()));
//    }

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);

    VectorTools::interpolate_boundary_values (mapping, dof_handler,
                                              0,
                                              BoundaryValues<dim>(),
                                              constraints);
    VectorTools::interpolate_boundary_values (mapping, dof_handler,
                                              1,
                                              BoundaryValues<dim>(),
                                              constraints);
    constraints.close ();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit (sparsity_pattern);


    // Setup GMG DoFs
    dof_handler.distribute_mg_dofs ();

    for (unsigned int level=0; level < triangulation.n_levels(); ++level)
    {
        if (settings.dof_renum == "random")
        {
            if (settings.smoother_type =="sor" ||
                    settings.smoother_type =="jacobi")
                Assert(false,ExcMessage("Random renumbering for point-smoothers not yet implemented."));
        }
        else if (settings.dof_renum == "downstream")
        {
            const AdvectionField<dim> w;
            DoFRenumbering::downstream(dof_handler,level,w.value(Point<dim>()));
        }
        else if (settings.dof_renum == "upstream")
        {
            const AdvectionField<dim> w;
            DoFRenumbering::downstream(dof_handler,level,-1.0*w.value(Point<dim>()));
        }
    }

    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof_handler);

    std::set<types::boundary_id>  dirichlet_boundary_ids;
    dirichlet_boundary_ids.insert(0);
    dirichlet_boundary_ids.insert(1);

    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary_ids);

    const unsigned int n_levels = triangulation.n_levels();

    mg_matrices.resize(0, n_levels-1);
    mg_matrices.clear_elements ();
    mg_interface_in.resize(0, n_levels-1);
    mg_interface_in.clear_elements ();
    mg_interface_out.resize(0, n_levels-1);
    mg_interface_out.clear_elements ();
    mg_sparsity_patterns.resize(0, n_levels-1);
    mg_interface_sparsity_patterns.resize(0, n_levels-1);

    for (unsigned int level=0; level<n_levels; ++level)
    {
        {
            DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                       dof_handler.n_dofs(level));
            MGTools::make_sparsity_pattern(dof_handler, dsp, level);
            mg_sparsity_patterns[level].copy_from (dsp);
            mg_matrices[level].reinit(mg_sparsity_patterns[level]);
        }
        {
            DynamicSparsityPattern dsp(dof_handler.n_dofs(level),
                                       dof_handler.n_dofs(level));
            MGTools::make_interface_sparsity_pattern(dof_handler, mg_constrained_dofs, dsp, level);
            mg_interface_sparsity_patterns[level].copy_from(dsp);
            mg_interface_in[level].reinit(mg_interface_sparsity_patterns[level]);
            mg_interface_out[level].reinit(mg_interface_sparsity_patterns[level]);
        }
    }
}


template <int dim>
template <class IteratorType>
void
AdvectionProblem<dim>::assemble_cell(const IteratorType &cell,
                                     ScratchData<dim> &  scratch_data,
                                     CopyData &          copy_data)
{
    const unsigned int level = cell->level();
    copy_data.level          = level;

    const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
    copy_data.dofs_per_cell = dofs_per_cell;

    const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();
    copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);

    if (!cell->is_level_cell())
        copy_data.cell_rhs.reinit(dofs_per_cell);

    copy_data.local_dof_indices.resize(dofs_per_cell);
    cell->get_active_or_mg_dof_indices(copy_data.local_dof_indices);

    scratch_data.fe_values.reinit(cell);

    Epsilon<dim>          epsilon;
    const AdvectionField<dim>   advection_field;
    const RightHandSide<dim>    right_hand_side;

    std::vector<double>         eps (n_q_points);
    std::vector<double>         rhs_values (n_q_points);
    std::vector<Tensor<1,dim> > advection_directions (n_q_points);

    epsilon.value_list (scratch_data.fe_values.get_quadrature_points(),
                        eps);
    advection_field.value_list (scratch_data.fe_values.get_quadrature_points(),
                                advection_directions);
    right_hand_side.value_list (scratch_data.fe_values.get_quadrature_points(),
                                rhs_values);

    double delta = 0.0;
    if (settings.with_sd)
        delta = delta_value(cell->diameter(),
                            epsilon.value(cell->center()),
                            advection_field.value(cell->center()),
                            settings.fe_degree);


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
                copy_data.cell_matrix(i, j) +=
                        (eps[q_point] *
                         scratch_data.fe_values.shape_grad(j, q_point) *
                         scratch_data.fe_values.shape_grad(i, q_point) *
                         scratch_data.fe_values.JxW(q_point))
                        +
                        ((advection_directions[q_point]*scratch_data.fe_values.shape_grad(j,q_point))*
                         scratch_data.fe_values.shape_value(i,q_point))
                        *scratch_data.fe_values.JxW(q_point);

                if (settings.with_sd)
                    copy_data.cell_matrix(i,j) += delta*
                            (advection_directions[q_point]*scratch_data.fe_values.shape_grad(j,q_point))*
                            (advection_directions[q_point]*scratch_data.fe_values.shape_grad(i,q_point))
                            * scratch_data.fe_values.JxW(q_point)
                            +
                            -1.0*delta*eps[q_point]*trace(scratch_data.fe_values.shape_hessian(j,q_point))*
                            (advection_directions[q_point]*scratch_data.fe_values.shape_grad(i,q_point))
                            * scratch_data.fe_values.JxW(q_point);
            }
            if (!cell->is_level_cell())
            {
                copy_data.cell_rhs(i) += scratch_data.fe_values.shape_value(i,q_point)*
                        rhs_values[q_point]
                        *scratch_data.fe_values.JxW (q_point);
                if (settings.with_sd)
                    copy_data.cell_rhs(i) += delta*
                            rhs_values[q_point]*
                            advection_directions[q_point]*scratch_data.fe_values.shape_grad(i,q_point)
                            *scratch_data.fe_values.JxW (q_point);
            }
        }
}


template <int dim>
void
AdvectionProblem<dim>::assemble_system_and_multigrid()
{
    auto cell_worker_active =
            [&](const decltype(dof_handler.begin_active()) &cell,
            ScratchData<dim> &                             scratch_data,
            CopyData &                                     copy_data)
    {
        this->assemble_cell(cell, scratch_data, copy_data);
    };



    auto copier_active = [&](const CopyData &copy_data)
    {
        constraints.distribute_local_to_global(copy_data.cell_matrix,
                                               copy_data.cell_rhs,
                                               copy_data.local_dof_indices,
                                               system_matrix,
                                               system_rhs);
    };


    MeshWorker::mesh_loop(dof_handler.begin_active(),
                          dof_handler.end(),
                          cell_worker_active,
                          copier_active,
                          ScratchData<dim>(fe, quad_degree),
                          CopyData(),
                          MeshWorker::assemble_own_cells);


    // Assemble GMG
    std::vector<AffineConstraints<double>> boundary_constraints(triangulation.n_global_levels());
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
    {
        IndexSet dofset;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                      level,
                                                      dofset);
        boundary_constraints[level].reinit(dofset);
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_refinement_edge_indices(level));
        boundary_constraints[level].add_lines(mg_constrained_dofs.get_boundary_indices(level));
        boundary_constraints[level].close();
    }

    auto cell_worker_mg = [&](const decltype(dof_handler.begin_mg()) &cell,
            ScratchData<dim>                                         &scratch_data,
            CopyData                                                 &copy_data)
    {
        this->assemble_cell(cell, scratch_data, copy_data);
    };

    auto copier_mg = [&](const CopyData &copy_data)
    {
        boundary_constraints[copy_data.level].distribute_local_to_global(copy_data.cell_matrix,
                                                                         copy_data.local_dof_indices,
                                                                         mg_matrices[copy_data.level]);

        // Will require explanation.
        for (unsigned int i = 0; i < copy_data.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < copy_data.dofs_per_cell; ++j)
                if (mg_constrained_dofs.is_interface_matrix_entry(copy_data.level,
                                                                  copy_data.local_dof_indices[i],
                                                                  copy_data.local_dof_indices[j]))
                {
                    mg_interface_out[copy_data.level].add(copy_data.local_dof_indices[i],
                                                          copy_data.local_dof_indices[j],
                                                          copy_data.cell_matrix(i,j));
                    mg_interface_in[copy_data.level].add(copy_data.local_dof_indices[i],
                                                         copy_data.local_dof_indices[j],
                                                         copy_data.cell_matrix(j,i));
                }
    };

    MeshWorker::mesh_loop(dof_handler.begin_mg(),
                          dof_handler.end_mg(),
                          cell_worker_mg,
                          copier_mg,
                          ScratchData<dim>(fe, quad_degree),
                          CopyData(),
                          MeshWorker::assemble_own_cells);
}


template <int dim>
std::shared_ptr<MGSmoother<Vector<double>>>
AdvectionProblem<dim>::create_smoother ()
{
    
    if (settings.smoother_type == "sor")
    {
        typedef PreconditionSOR<SparseMatrix<double> > Smoother;
		
        auto smoother = std::make_shared<MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >>();
	smoother->initialize(mg_matrices,Smoother::AdditionalData(fe.degree == 1 ? 1.0 : 0.75));
        smoother->set_steps(2);
	return smoother;
    }
    else if (settings.smoother_type == "jacobi")
    {
        typedef PreconditionJacobi<SparseMatrix<double> > Smoother;
        auto smoother = std::make_shared<MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >>();
        smoother->initialize(mg_matrices, Smoother::AdditionalData(fe.degree == 1 ? 0.6667 : 0.47));
        smoother->set_steps(4);
        return smoother;
    }
    else if (settings.smoother_type == "block sor")
    {
        typedef RelaxationBlockSOR<SparseMatrix<double>, double, Vector<double> > Smoother;

        static MGLevelObject<typename Smoother::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_levels() - 1);

        for (unsigned int level=0; level < triangulation.n_levels(); ++level)
        {
            DoFTools::make_cell_patches(smoother_data[level].block_list, dof_handler, level);

            smoother_data[level].relaxation = 1.0;
            smoother_data[level].inversion = PreconditionBlockBase<double>::svd;

            if (settings.dof_renum == "downstream")
            {
                const AdvectionField<dim>   advection_field;
                std::vector<unsigned int> ordered_indices =
                        create_downstream_order(dof_handler,
                                                advection_field.value(triangulation.begin()->center()),
                                                level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
            else if (settings.dof_renum == "upstream")
            {
                const AdvectionField<dim>   advection_field;
                std::vector<unsigned int> ordered_indices =
                        create_downstream_order(dof_handler,
                                                -1.0*advection_field.value(triangulation.begin()->center()),
                                                level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
            else if (settings.dof_renum == "random")
            {
                std::vector<unsigned int> ordered_indices =
                        create_random_order(dof_handler, level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
        }

	auto smoother = std::make_shared<MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >>();
	
        smoother->initialize(mg_matrices, smoother_data);
        smoother->set_steps(1);
        return smoother;
    }
    else if (settings.smoother_type == "block jacobi")
    {
        typedef RelaxationBlockJacobi<SparseMatrix<double>, double, Vector<double> > Smoother;

        static MGLevelObject<typename Smoother::AdditionalData> smoother_data;
        smoother_data.resize(0, triangulation.n_levels() - 1);

        for (unsigned int level=0; level < triangulation.n_levels(); ++level)
        {
            DoFTools::make_cell_patches(smoother_data[level].block_list, dof_handler, level);

            smoother_data[level].relaxation = 0.25;
            smoother_data[level].inversion = PreconditionBlockBase<double>::svd;

            if (settings.dof_renum == "downstream")
            {
                const AdvectionField<dim>   advection_field;
                std::vector<unsigned int> ordered_indices =
                        create_downstream_order(dof_handler,
                                                advection_field.value(triangulation.begin()->center()),
                                                level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
            else if (settings.dof_renum == "upstream")
            {
                const AdvectionField<dim>   advection_field;
                std::vector<unsigned int> ordered_indices =
                        create_downstream_order(dof_handler,
                                                -1.0*advection_field.value(triangulation.begin()->center()),
                                                level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
            else if (settings.dof_renum == "random")
            {
                std::vector<unsigned int> ordered_indices =
                        create_random_order(dof_handler, level);
                smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
            }
        }

	auto smoother = std::make_shared<MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >>();
        smoother->initialize(mg_matrices, smoother_data);
        smoother->set_steps(2);
        return smoother;
    }
    else
      AssertThrow(false, ExcNotImplemented());
}

  
template <int dim>
void AdvectionProblem<dim>::solve ()
{
    Timer time;

    double solve_tol = 1e-8*system_rhs.l2_norm();
    unsigned int max_iters = 200;
    SolverControl solver_control (max_iters, solve_tol, true, true);
    solver_control.enable_history_data();



    typedef MGTransferPrebuilt<Vector<double> > Transfer;
    Transfer mg_transfer(mg_constrained_dofs);
    mg_transfer.build_matrices(dof_handler);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from (mg_matrices[0]);
    MGCoarseGridHouseholder<double, Vector<double> > coarse_grid_solver;
    coarse_grid_solver.initialize (coarse_matrix);

    std::shared_ptr<MGSmoother<Vector<double>>> mg_smoother = create_smoother();

    mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
    mg::Matrix<Vector<double> > mg_interface_matrix_in(mg_interface_in);
    mg::Matrix<Vector<double> > mg_interface_matrix_out(mg_interface_out);

    Multigrid<Vector<double> > mg(mg_matrix,
                                  coarse_grid_solver,
                                  mg_transfer,
                                  *mg_smoother,
                                  *mg_smoother);
    mg.set_edge_matrices(mg_interface_matrix_out, mg_interface_matrix_in);

    PreconditionMG<dim, Vector<double>, Transfer>
            preconditioner(dof_handler, mg, mg_transfer);


    std::cout << "     Solving with GMRES to tol " << solve_tol << "..." << std::endl;
    SolverGMRES<>    solver (solver_control);

    time.restart();
    solver.solve (system_matrix, solution, system_rhs,
                  preconditioner);
    time.stop();

    std::cout << "          converged in " << solver_control.last_step() << " iterations"
	      << " in " << time.last_wall_time()
              << " seconds " << std::endl;

    constraints.distribute (solution);
}



// Just for pictures
template <int dim>
void AdvectionProblem<dim>::output_results (const unsigned int cycle) const
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");

    Vector<double> cell_indices (triangulation.n_active_cells());
    if (settings.dof_renum == "downstream")
    {
        const AdvectionField<dim>   advection_field;
        std::vector<unsigned int> ordered_indices =
                create_downstream_order(dof_handler,
                                        advection_field.value(triangulation.begin_active()->center()));
        for (unsigned int i=0; i<ordered_indices.size(); ++i)
            cell_indices(ordered_indices[i]) = i;
    }
    else if (settings.dof_renum == "upstream")
    {
        const AdvectionField<dim>   advection_field;
        std::vector<unsigned int> ordered_indices =
                create_downstream_order(dof_handler,
                                        -1.0*advection_field.value(triangulation.begin_active()->center()));
        for (unsigned int i=0; i<ordered_indices.size(); ++i)
            cell_indices(ordered_indices[i]) = i;
    }
    else if (settings.dof_renum == "no")
    {
        typename DoFHandler<dim>::active_cell_iterator
                cell=dof_handler.begin_active(),
                endc=dof_handler.end();
        for (; cell!=endc; ++cell)
        {
            cell_indices(cell->index()) = cell->index();
        }
    }
    else if (settings.dof_renum == "random")
    {
        std::vector<unsigned int> ordered_indices =
                create_random_order(dof_handler);
        for (unsigned int i=0; i<ordered_indices.size(); ++i)
            cell_indices(ordered_indices[i]) = i;
    }
    data_out.add_data_vector (cell_indices, "cell_indx");

    data_out.build_patches ();
    {
        std::ostringstream filename;
        filename << "solution-"
                 << cycle
                 << ".vtu";
        std::ofstream output (filename.str().c_str());
        data_out.write_vtu (output);
    }
}


template <int dim>
void AdvectionProblem<dim>::run ()
{
    for (unsigned int cycle=0; cycle < (settings.fe_degree == 1 ? 7 : 5); ++cycle)
    {
        std::cout << "  Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube_with_cylindrical_hole	(triangulation,0.3,1.0,
                                                             0.5,1,false);
            static const SphericalManifold<dim> manifold_description(Point<dim>(0,0));
            triangulation.set_manifold (1, manifold_description);

            triangulation.refine_global (1);

        }
        else
        {
            triangulation.refine_global();
        }

        setup_system ();

        std::cout << "     Number of active cells:       "
                  << triangulation.n_active_cells()
                  << " (" << triangulation.n_levels() << " levels)"
                  << std::endl;
        std::cout << "     Number of degrees of freedom: "
                  << dof_handler.n_dofs()
                  << std::endl;

        assemble_system_and_multigrid ();

        solve ();

        if (settings.output)
            output_results (cycle);

        std::cout << std::endl;
    }
}
}


int main (int argc, char *argv[])
{
    try
    {
        Step100::Settings settings;
        if (!settings.try_parse((argc>1) ? (argv[1]) : ""))
            return 0;

        Step100::AdvectionProblem<2> advection_problem_2d(settings);
        advection_problem_2d.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
