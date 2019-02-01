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
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
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



namespace FinalProject
{
using namespace dealii;


struct Settings
{
  bool try_parse(const std::string &prm_filename);

  std::string ref_type;
  unsigned int initial_ref;
  double epsilon;
  unsigned int fe_degree;

  std::string smooth_solve_type;

  bool smoothing_and_damping_overwrite;
  unsigned int smooth_steps;
  double relax;

  std::string dof_renum;
  bool with_sd;
  bool full_boundary;

  unsigned int n_cycles;
  bool output;
};
bool
Settings::try_parse(const std::string &prm_filename)
{
  ParameterHandler prm;

  prm.declare_entry("ref type", "global",
                    Patterns::Selection("global|adaptive"),
                    "Refinment Type: global|adaptive");
  prm.declare_entry("initial ref", "4",
                    Patterns::Integer(0),
                    "Number of initial refinements");
  prm.declare_entry("epsilon", "0.005",
                    Patterns::Double(0.0),
                    "Epsilon");
  prm.declare_entry("fe degree", "1",
                    Patterns::Integer(0),
                    "Finite Element degree");


  prm.declare_entry("smoother or solver type", "direct",
                    Patterns::Selection("direct|sor|jacobi|block_sor|block_jacobi"),
                    "Smoother Type: direct|sor|jacobi|block_sor|block_jacobi");

  prm.declare_entry("overwrite smoothing and damping", "false",
                    Patterns::Bool(),
                    "Overwrite number of smoothing steps and damping parameter: true|false");
  prm.declare_entry("smoothing steps", "1",
                    Patterns::Integer(0),
                    "Number of smoothing steps");
  prm.declare_entry("relaxation", "0.8",
                    Patterns::Double(0.0),
                    "Relaxation parameter");


  prm.declare_entry("dof renumbering", "no",
                    Patterns::Selection("no|random|downstream|upstream"),
                    "Dof renumbering: no|random|downstream|upstream");

  prm.declare_entry("with sd", "false",
                    Patterns::Bool(),
                    "With streamline diffusion: true|false");

  prm.declare_entry("full boundary", "true",
                    Patterns::Bool(),
                    "Full boundary Dirichlet: true|false");

  prm.declare_entry("number of cycles", "0",
                    Patterns::Integer(0),
                    "Number of cycles");
  prm.declare_entry("output", "false",
                    Patterns::Bool(),
                    "Output: true|false");
  try
  {
    prm.parse_input(prm_filename);
  }
  catch (...)
  {
    prm.print_parameters(std::cout, ParameterHandler::Text);
    return false;
  }
  this->ref_type = prm.get("ref type");
  this->initial_ref = prm.get_integer("initial ref");
  this->epsilon = prm.get_double("epsilon");
  this->fe_degree = prm.get_integer("fe degree");

  this->smooth_solve_type = prm.get("smoother or solver type");

  this->smoothing_and_damping_overwrite = prm.get_bool("overwrite smoothing and damping");
  this->smooth_steps = prm.get_integer("smoothing steps");
  this->relax = prm.get_double("relaxation");

  this->dof_renum = prm.get("dof renumbering");

  this->with_sd = prm.get_bool("with sd");

  this->full_boundary = prm.get_bool("full boundary");

  this->n_cycles = prm.get_integer("number of cycles");
  this->output = prm.get_bool("output");

  return true;
}



namespace Conrad
{
unsigned int n_smooths (Settings settings)
{
  if (settings.smooth_solve_type == "sor")
  {
    return 2;
  }
  else if (settings.smooth_solve_type == "jacobi")
  {
    return 4;
  }
  else if (settings.smooth_solve_type == "block_sor")
  {
    return 1;
  }
  else if (settings.smooth_solve_type == "block_jacobi")
  {
    return 2;
  }
  return 1;
}
double relax_param (Settings settings)
{
  if (settings.smooth_solve_type == "sor")
  {
    if (settings.fe_degree == 1)
      return 1.0;
    else if (settings.fe_degree == 3)
      return 0.75;
    else
      Assert(settings.smoothing_and_damping_overwrite,
             ExcMessage("Defaults only for degree 1 and 3"));
  }
  else if (settings.smooth_solve_type == "jacobi")
  {
    if (settings.fe_degree == 1)
      return 0.6667;
    else if (settings.fe_degree == 3)
      return 0.47;
    else
      Assert(settings.smoothing_and_damping_overwrite,
             ExcMessage("Defaults only for degree 1 and 3"));
  }
  else if (settings.smooth_solve_type == "block_sor")
  {
    return 1.0;
  }
  else if (settings.smooth_solve_type == "block_jacobi")
  {
    return 0.25;
  }
  return 1.0;
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

template <int dim>
std::vector<unsigned int>
create_downstream_order_level (const DoFHandler<dim> &dof,
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
create_downstream_order_active (const DoFHandler<dim> &dof,
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
create_random_order_level (const DoFHandler<dim> &dof,
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
create_random_order_active (const DoFHandler<dim> &dof)
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

  virtual void get_settings (const Settings set);
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;

  Settings settings;
};
template <int dim>
void Epsilon<dim>::get_settings(const Settings set)
{
  settings = set;
}
template <int dim>
double Epsilon<dim>::value (const Point<dim> &p,
                            const unsigned int) const
{
  (void)p;
  return settings.epsilon;
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
  else if (std::fabs(p[0]-1)<1e-8
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
  //std::cosh(Peclet)/std::sinh(Peclet);

  return hk/(2.0*dir.norm()*pk)*(coth - 1.0/Peclet);
}


template <int dim>
class AdvectionProblem
{
public:
  AdvectionProblem (Settings settings);
  ~AdvectionProblem ();
  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void assemble_multigrid ();
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

  bool solve_complete;
};






template <int dim>
AdvectionProblem<dim>::AdvectionProblem (Settings settings)
  :
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
    dof_handler (triangulation),
    fe(settings.fe_degree),
    mapping(settings.fe_degree),
    quad_degree (2*fe.degree+2),
    settings(settings),
    solve_complete(true)
{}



template <int dim>
AdvectionProblem<dim>::~AdvectionProblem ()
{
  dof_handler.clear ();
}



template <int dim>
void AdvectionProblem<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  if (settings.dof_renum == "random")
  {
    if (settings.smooth_solve_type =="direct" ||
        settings.smooth_solve_type =="sor" ||
        settings.smooth_solve_type =="jacobi")
      Assert(false,ExcMessage("Random renumbering for point-smoothers not yet implemented."));
  }
  else if (settings.dof_renum == "downstream")
  {
    const AdvectionField<dim> w;
    DoFRenumbering::downstream(dof_handler,w.value(Point<dim>()));
  }
  else if (settings.dof_renum == "upstream")
  {
    const AdvectionField<dim> w;
    DoFRenumbering::downstream(dof_handler,-1.0*w.value(Point<dim>()));
  }

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);

  VectorTools::interpolate_boundary_values (mapping, dof_handler,
                                            1,
                                            BoundaryValues<dim>(),
                                            constraints);
  if (settings.full_boundary)
    VectorTools::interpolate_boundary_values (mapping, dof_handler,
                                              4,
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


  // GMG Specific
  if (settings.smooth_solve_type != "direct")
  {
    dof_handler.distribute_mg_dofs ();

    for (unsigned int level=0; level < triangulation.n_levels(); ++level)
    {
      if (settings.dof_renum == "random")
      {
        if (settings.smooth_solve_type =="sor" ||
            settings.smooth_solve_type =="jacobi")
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
    dirichlet_boundary_ids.insert(1);
    if (settings.full_boundary)
      dirichlet_boundary_ids.insert(4);

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

    mg_constraints.resize (0, n_levels-1);

    std::vector<IndexSet> boundary_indices(triangulation.n_levels());
    MGTools::make_boundary_list (dof_handler,
                                 dirichlet_boundary_ids,
                                 boundary_indices);

    for (unsigned int level=0; level<n_levels; ++level)
    {
      IndexSet::ElementIterator bc_it = boundary_indices[level].begin();
      for ( ; bc_it != boundary_indices[level].end(); ++bc_it)
        mg_constraints[level].add_line(*bc_it);
      mg_constraints[level].close();

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
}



template <int dim>
void AdvectionProblem<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(quad_degree);

  FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                           update_values    |  update_gradients | update_hessians |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Epsilon<dim>          epsilon;
  epsilon.get_settings(settings);
  const AdvectionField<dim>   advection_field;
  const RightHandSide<dim>    right_hand_side;
  std::vector<double>         eps (n_q_points);
  std::vector<double>         rhs_values (n_q_points);
  std::vector<Tensor<1,dim> > advection_directions (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit (cell);

    epsilon.value_list (fe_values.get_quadrature_points(),
                        eps);
    advection_field.value_list (fe_values.get_quadrature_points(),
                                advection_directions);
    right_hand_side.value_list (fe_values.get_quadrature_points(),
                                rhs_values);

    double delta = 0.0;
    if (settings.with_sd)
      delta = delta_value(cell->diameter(),
                          epsilon.value(cell->center()),
                          advection_field.value(cell->center()),
                          settings.fe_degree);

    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          cell_matrix(i,j) += (eps[q]*(fe_values.shape_grad (j, q)*
                                       fe_values.shape_grad (i, q))
                               *fe_values.JxW(q)
                               +
                               ((advection_directions[q]*fe_values.shape_grad(j,q))*
                                fe_values.shape_value(i,q))
                               *fe_values.JxW(q));
          if (settings.with_sd)
            cell_matrix(i,j) += (delta*
                                 (advection_directions[q]*fe_values.shape_grad(j,q))*
                                 (advection_directions[q]*fe_values.shape_grad(i,q))
                                 * fe_values.JxW(q)
                                 +
                                 -1.0*delta*eps[q]*trace(fe_values.shape_hessian(j,q))*
                                 (advection_directions[q]*fe_values.shape_grad(i,q))
                                 * fe_values.JxW(q));
        }

        cell_rhs(i) += (fe_values.shape_value(i,q)*
                        rhs_values[q])
            *fe_values.JxW (q);
        if (settings.with_sd)
          cell_rhs(i) += delta*
              rhs_values[q]*
              advection_directions[q]*fe_values.shape_grad(i,q)
              *fe_values.JxW (q);
      }

    cell->get_dof_indices (local_dof_indices);
    constraints.distribute_local_to_global (cell_matrix, cell_rhs,
                                            local_dof_indices,
                                            system_matrix, system_rhs);
  }
}


template <int dim>
void AdvectionProblem<dim>::assemble_multigrid ()
{
  const QGauss<dim>  quadrature_formula(quad_degree);

  FEValues<dim> fe_values (mapping, fe, quadrature_formula,
                           update_values    |  update_gradients | update_hessians |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int n_levels = triangulation.n_levels();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Epsilon<dim>          epsilon;
  epsilon.get_settings(settings);
  const AdvectionField<dim>   advection_field;
  std::vector<double>         eps (n_q_points);
  std::vector<Tensor<1,dim> > advection_directions (n_q_points);

  std::vector<AffineConstraints<double>> boundary_constraints (n_levels);
  for (unsigned int level=0; level<n_levels; ++level)
  {
    IndexSet dofset;
    DoFTools::extract_locally_relevant_level_dofs (dof_handler, level, dofset);
    boundary_constraints[level].reinit(dofset);
    boundary_constraints[level].add_lines (mg_constrained_dofs.get_refinement_edge_indices(level));
    boundary_constraints[level].add_lines (mg_constrained_dofs.get_boundary_indices(level));
    boundary_constraints[level].close ();
  }

  typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(),
      endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell_matrix = 0;

    fe_values.reinit (cell);
    const unsigned int level = cell->level();

    epsilon.value_list (fe_values.get_quadrature_points(),
                        eps);
    advection_field.value_list (fe_values.get_quadrature_points(),
                                advection_directions);


    double delta = 0.0;
    if (settings.with_sd)
      delta = delta_value(cell->diameter(),
                          epsilon.value(cell->center()),
                          advection_field.value(cell->center()),
                          settings.fe_degree);

    for (unsigned int q=0; q<n_q_points; ++q)
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          cell_matrix(i,j) += (eps[q]*(fe_values.shape_grad (j, q)*
                                       fe_values.shape_grad (i, q))
                               *fe_values.JxW(q)
                               +
                               ((advection_directions[q]*fe_values.shape_grad(j,q))*
                                fe_values.shape_value(i,q))
                               *fe_values.JxW(q));
          if (settings.with_sd)
            cell_matrix(i,j) += (delta*
                                 (advection_directions[q]*fe_values.shape_grad(j,q))*
                                 (advection_directions[q]*fe_values.shape_grad(i,q))
                                 * fe_values.JxW(q)
                                 +
                                 -1.0*delta*eps[q]*trace(fe_values.shape_hessian(j,q))*
                                 (advection_directions[q]*fe_values.shape_grad(i,q))
                                 * fe_values.JxW(q));
        }
    cell->get_mg_dof_indices (local_dof_indices);
    boundary_constraints[level]
        .distribute_local_to_global (cell_matrix,
                                     local_dof_indices,
                                     mg_matrices[level]);


    // EXPLAIN
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        if (mg_constrained_dofs.is_interface_matrix_entry(level, local_dof_indices[i], local_dof_indices[j]))
        {
          mg_interface_out[level].add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
          mg_interface_in[level].add(local_dof_indices[i],local_dof_indices[j],cell_matrix(j,i));
        }
  }
}

template <int dim>
void AdvectionProblem<dim>::solve ()
{
  Timer time;

  if (!settings.smoothing_and_damping_overwrite)
  {
    settings.smooth_steps = Conrad::n_smooths(settings);
    settings.relax = Conrad::relax_param(settings);
  }

  double solve_tol = 1e-8*system_rhs.l2_norm();
  unsigned int max_iters = 200;
  SolverControl solver_control (max_iters, solve_tol, true, true);
  solver_control.enable_history_data();


  if (settings.smooth_solve_type == "direct")
  {
    SparseDirectUMFPACK direct_solver;

    Vector<double> tmp_rhs;
    tmp_rhs = system_rhs;
    time.restart();
    direct_solver.solve (system_matrix,tmp_rhs);
    time.stop();

    solution = tmp_rhs;
    constraints.distribute (solution);
    std::cout << "     Direct Solver: " << time.last_wall_time() << " seconds" << std::endl;
  }
  else
  {
    typedef MGTransferPrebuilt<Vector<double> > Transfer;
    Transfer mg_transfer(mg_constrained_dofs);
    mg_transfer.build_matrices(dof_handler);

    FullMatrix<double> coarse_matrix;
    coarse_matrix.copy_from (mg_matrices[0]);
    MGCoarseGridHouseholder<double, Vector<double> > coarse_grid_solver;
    coarse_grid_solver.initialize (coarse_matrix);


    MGSmoother<Vector<double> > *mg_smoother = NULL;
    if (settings.smooth_solve_type == "sor")
    {
      typedef PreconditionSOR<SparseMatrix<double> > Smoother;

      auto *smoother =
          new MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >();

      smoother->initialize(mg_matrices,Smoother::AdditionalData(settings.relax));
      mg_smoother = smoother;
    }
    else if (settings.smooth_solve_type == "jacobi")
    {
      typedef PreconditionJacobi<SparseMatrix<double> > Smoother;
      auto *smoother =
          new MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >();
      smoother->initialize(mg_matrices, Smoother::AdditionalData(settings.relax));
      mg_smoother = smoother;
    }
    else if (settings.smooth_solve_type == "block_sor")
    {
      typedef RelaxationBlockSOR<SparseMatrix<double>, double, Vector<double> > Smoother;

      static MGLevelObject<typename Smoother::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_levels() - 1);

      for (unsigned int level=0; level < triangulation.n_levels(); ++level)
      {
        DoFTools::make_cell_patches(smoother_data[level].block_list, dof_handler, level);

        smoother_data[level].relaxation = settings.relax;
        smoother_data[level].inversion = PreconditionBlockBase<double>::svd;

        if (settings.dof_renum == "downstream")
        {
          const AdvectionField<dim>   advection_field;
          std::vector<unsigned int> ordered_indices =
              Conrad::create_downstream_order_level(dof_handler,
                                                    advection_field.value(triangulation.begin()->center()),
                                                    level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
        else if (settings.dof_renum == "upstream")
        {
          const AdvectionField<dim>   advection_field;
          std::vector<unsigned int> ordered_indices =
              Conrad::create_downstream_order_level(dof_handler,
                                                    -1.0*advection_field.value(triangulation.begin()->center()),
                                                    level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
        else if (settings.dof_renum == "random")
        {
          std::vector<unsigned int> ordered_indices =
              Conrad::create_random_order_level(dof_handler, level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
      }

      auto *smoother =
          new MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >();
      smoother->initialize(mg_matrices, smoother_data);
      mg_smoother = smoother;
    }
    else if (settings.smooth_solve_type == "block_jacobi")
    {
      typedef RelaxationBlockJacobi<SparseMatrix<double>, double, Vector<double> > Smoother;

      static MGLevelObject<typename Smoother::AdditionalData> smoother_data;
      smoother_data.resize(0, triangulation.n_levels() - 1);

      for (unsigned int level=0; level < triangulation.n_levels(); ++level)
      {
        DoFTools::make_cell_patches(smoother_data[level].block_list, dof_handler, level);

        smoother_data[level].relaxation = settings.relax;
        smoother_data[level].inversion = PreconditionBlockBase<double>::svd;

        if (settings.dof_renum == "downstream")
        {
          const AdvectionField<dim>   advection_field;
          std::vector<unsigned int> ordered_indices =
              Conrad::create_downstream_order_level(dof_handler,
                                                    advection_field.value(triangulation.begin()->center()),
                                                    level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
        else if (settings.dof_renum == "upstream")
        {
          const AdvectionField<dim>   advection_field;
          std::vector<unsigned int> ordered_indices =
              Conrad::create_downstream_order_level(dof_handler,
                                                    -1.0*advection_field.value(triangulation.begin()->center()),
                                                    level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
        else if (settings.dof_renum == "random")
        {
          std::vector<unsigned int> ordered_indices =
              Conrad::create_random_order_level(dof_handler, level);
          smoother_data[level].order = std::vector<std::vector<unsigned int> > (1, ordered_indices);
        }
      }

      auto *smoother =
          new MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> >();
      smoother->initialize(mg_matrices, smoother_data);
      mg_smoother = smoother;
    }




    mg_smoother->set_steps(settings.smooth_steps);
    //mg_smoother->set_debug(10);

    mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
    mg::Matrix<Vector<double> > mg_interface_matrix_in(mg_interface_in);
    mg::Matrix<Vector<double> > mg_interface_matrix_out(mg_interface_out);

    Multigrid<Vector<double> > mg(mg_matrix,
                                  coarse_grid_solver,
                                  mg_transfer,
                                  *mg_smoother,
                                  *mg_smoother);
    mg.set_edge_matrices(mg_interface_matrix_out, mg_interface_matrix_in);

    //mg.set_debug(10);

    PreconditionMG<dim, Vector<double>, Transfer>
        preconditioner(dof_handler, mg, mg_transfer);


    std::cout << "     Solving with GMRES tol " << solve_tol << "..." << std::endl;
    SolverGMRES<>    solver (solver_control);


    mg_smoother->set_steps(settings.smooth_steps);
    try
    {
      solution *= 0;
      time.restart();
      solver.solve (system_matrix, solution, system_rhs,
                    preconditioner);
      time.stop();

      std::cout << "     GMRES Solver: " << time.last_wall_time()
                << " seconds " << std::endl
                << "          converged in " << solver_control.last_step() << " iterations" << std::endl;
    }
    catch (SolverControl::NoConvergence)
    {
      solve_complete = false;

      std::cout << "********************************************************************" << std::endl
                << "SOLVER DID NOT CONVERGE AFTER "
                << solver_control.last_step()
                << " ITERATIONS. res=" << solver_control.last_value() << std::endl
                << "********************************************************************" << std::endl;
    }

    constraints.distribute (solution);

    mg_smoother->clear();
  }

  return;
}


template <int dim>
void AdvectionProblem<dim>::refine_grid ()
{
  if (settings.ref_type == "global")
  {
    triangulation.refine_global();
  }
  else if (settings.ref_type == "adaptive")
  {
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate (mapping,
                                        dof_handler,
                                        QGauss<dim-1>(quad_degree),
                                        std::map<types::boundary_id, const Function<dim> *>(),
                                        solution,
                                        estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     estimated_error_per_cell,
                                                     0.3333,0.0);
    triangulation.execute_coarsening_and_refinement ();
  }
}



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
        Conrad::create_downstream_order_active(dof_handler,
                                               advection_field.value(triangulation.begin_active()->center()));
    for (unsigned int i=0; i<ordered_indices.size(); ++i)
      cell_indices(ordered_indices[i]) = i;
  }
  else if (settings.dof_renum == "upstream")
  {
    const AdvectionField<dim>   advection_field;
    std::vector<unsigned int> ordered_indices =
        Conrad::create_downstream_order_active(dof_handler,
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
        Conrad::create_random_order_active(dof_handler);
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

  //  GridOut grid_out;
  //  grid_out.write_mesh_per_processor_as_vtu(triangulation,
  //                                           "grid-"+Utilities::int_to_string(cycle),
  //                                           true);
}


template <int dim>
void AdvectionProblem<dim>::run ()
{
  std::string problem_str = "epsilon=" + Utilities::to_string(settings.epsilon) + ", "
      + (settings.smooth_solve_type!="direct" ? settings.smooth_solve_type+" smoother, " : "direct solver, ")
      + (settings.dof_renum + " renumbering, ")
      + (settings.with_sd ? "with streamline diffusion, " : "without streamline diffusion, ")
      + "Q" + Utilities::int_to_string(settings.fe_degree) + " element.";
  std::cout << problem_str << std::endl;

  for (unsigned int cycle=0; cycle<settings.n_cycles; ++cycle)
  {
    std::cout << "  Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      GridGenerator::hyper_cube_with_cylindrical_hole	(triangulation,0.3,1.0,
                                                       0.5,1,true);
      static const SphericalManifold<dim> manifold_description(Point<dim>(0,0));
      triangulation.set_manifold (4, manifold_description);

      if (settings.full_boundary)
      {
        typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
            endc = triangulation.end();
        for (;cell!=endc;++cell)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() != 4)
              cell->face(f)->set_boundary_id(1);
      }
      else
      {
        typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
            endc = triangulation.end();
        for (;cell!=endc;++cell)
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
            {
              if (std::fabs(cell->face(f)->center()[0]-1)<1e-8 ||
                  std::fabs(cell->face(f)->center()[1]-(-1))<1e-8)
                cell->face(f)->set_boundary_id(1);
              else if (cell->face(f)->boundary_id() == 4)
                continue;
              else
                cell->face(f)->set_boundary_id(0);
            }
      }


      //      GridGenerator::hyper_cube(triangulation,0,1);
      //      typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
      //          endc = triangulation.end();
      //      for (;cell!=endc;++cell)
      //        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      //          if (cell->face(f)->at_boundary())
      //            cell->face(f)->set_boundary_id(1);


      triangulation.refine_global (settings.initial_ref);

    }
    else
    {
      refine_grid ();
    }

    setup_system ();

    std::cout << "     Number of active cells:       "
              << triangulation.n_active_cells()
              << " (" << triangulation.n_levels() << " levels)"
              << std::endl;
    std::cout << "     Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    assemble_system ();

    if (settings.smooth_solve_type != "direct")
      assemble_multigrid();

    solve ();

    if (settings.output)
      output_results (cycle);

    std::cout << std::endl;

    if (settings.ref_type == "global")
      if (dof_handler.n_dofs()*4 > 2e5)
        return;

    if (settings.ref_type == "adaptive")
      if (dof_handler.n_dofs()*2 > 5e4)
        return;
  }
}
}



int main (int argc, char *argv[])
{
  try
  {
    FinalProject::Settings settings;
    if (!settings.try_parse((argc>1) ? (argv[1]) : ""))
      return 0;

    //dealii::deallog.depth_console(10);

    FinalProject::AdvectionProblem<2> advection_problem_2d(settings);
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
