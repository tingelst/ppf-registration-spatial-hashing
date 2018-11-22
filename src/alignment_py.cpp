#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <pcl/common/angles.h>
#include <pcl/common/distances.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <vector_types.h>

#include "impl/cycle_iterator.hpp"
#include "impl/scene_generation.hpp"
#include "impl/util.hpp"
#include "kernel.h"
#include "linalg.h"
#include "ppf.h"
#include "vector_ops.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename Point>
typename pcl::PointCloud<Point>::Ptr loadPLYFile(const std::string& file_name)
{
  typename pcl::PointCloud<Point>::Ptr cloud(new pcl::PointCloud<Point>);
  pcl::io::loadPLYFile<Point>(file_name, *cloud);
  return cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr randomDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float p)
{
  pcl::RandomSample<Point> rs;
  typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
  rs.setSample(p * cloud->size());
  rs.setInputCloud(cloud);
  rs.filter(*filtered_cloud);
  return filtered_cloud;
}
template <typename Point>
typename pcl::PointCloud<Point>::Ptr sequentialDownsample(typename pcl::PointCloud<Point>::Ptr cloud, int n)
{
  typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
  int i = 0;
  for (typename pcl::PointCloud<Point>::iterator it = cloud->begin(); it != cloud->end(); ++it, ++i)
  {
    if (i % n == 0)
    {
      filtered_cloud->push_back(*it);
    }
  }
  return filtered_cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr voxelGridDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float leaf)
{
  typename pcl::PointCloud<Point>::Ptr filtered_cloud(new pcl::PointCloud<Point>);
  pcl::VoxelGrid<Point> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf, leaf, leaf);
  sor.filter(*filtered_cloud);
  return filtered_cloud;
}

namespace ppf_registration_spatial_hashing
{
using Point = pcl::PointNormal;
using PointCloud = pcl::PointCloud<pcl::PointNormal>;
}  // namespace ppf_registration_spatial_hashing

namespace ppf = ppf_registration_spatial_hashing;

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

PYBIND11_MODULE(ppf_registration_spatial_hashing, m)
{
  py::class_<ppf::Point>(m, "Point")
      .def_readonly("x", &ppf::Point::x)
      .def_readonly("y", &ppf::Point::y)
      .def_readonly("z", &ppf::Point::z)
      .def("__repr__", [](const ppf::Point& self) {
        std::stringstream ss;
        ss << self;
        return ss.str();
      });

  auto point_cloud = py::class_<ppf::PointCloud, ppf::PointCloud::Ptr>(m, "PointCloud")
                         .def_readonly("width", &ppf::PointCloud::width)
                         .def_readonly("height", &ppf::PointCloud::height)
                         .def_readonly("points", &ppf::PointCloud::points)
                         .def_readonly("sensor_origin", &ppf::PointCloud::sensor_origin_)
                         .def_readonly("sensor_orientation", &ppf::PointCloud::sensor_orientation_)
                         .def("array", [](const ppf::PointCloud& self) { return self.getMatrixXfMap(); });

  m.def("load_ply_file", &loadPLYFile<ppf::Point>);
  m.def("random_downsample", &randomDownsample<ppf::Point>);
  m.def("sequential_downsample", &sequentialDownsample<ppf::Point>);
  m.def("voxelgrid_downsample", &voxelGridDownsample<ppf::Point>);

  m.def("min_max_3d", [](const ppf::PointCloud& point_cloud) {
    pcl::PointNormal minPt, maxPt;
    // Finding the max pairwise distance is expensive, so
    // approximate it with the max difference between coords.
    pcl::getMinMax3D(point_cloud, minPt, maxPt);
    return std::make_tuple(minPt, maxPt);
  });

  m.def("transform_point_cloud", [](const ppf::PointCloud& point_cloud, const Eigen::Matrix4f& transform) {
    ppf::PointCloud transformed_point_cloud;
    pcl::transformPointCloudWithNormals(point_cloud, transformed_point_cloud, transform);
    return transformed_point_cloud;
  });

  m.def("icp", [](ppf::PointCloud::Ptr scene, ppf::PointCloud::Ptr model) {
    pcl::IterativeClosestPoint<ppf::Point, ppf::Point> icp;
    icp.setInputSource(model);
    icp.setInputTarget(scene);
    ppf::PointCloud final;
    icp.align(final);
    BOOST_LOG_TRIVIAL(info) << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore()
                            << std::endl;
    return icp.getFinalTransformation();
  });

  m.def("ppf_registration",
        [](pcl::PointCloud<pcl::PointNormal>::Ptr scene_cloud, pcl::PointCloud<pcl::PointNormal>::Ptr model_cloud,
           float model_d_dist, unsigned int ref_point_downsample_factor, float vote_count_threshold,
           bool cpu_clustering, bool use_l1_norm, bool use_averaged_clusters, int devUse) {
          Eigen::Matrix4f result;
          result = ppf_registration(std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>{ scene_cloud },
                                    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr>{ model_cloud },
                                    std::vector<float>{ model_d_dist }, ref_point_downsample_factor,
                                    vote_count_threshold, cpu_clustering, use_l1_norm, use_averaged_clusters, devUse);
          return result;
        },
        py::arg("scene_cloud"), py::arg("model_cloud"), py::arg("model_d_dist"), py::arg("ref_point_downsample_factor"),
        py::arg("vote_count_threshold"), py::arg("cpu_clustering") = false, py::arg("use_l1_norm") = false,
        py::arg("use_averaged_clusters") = false, py::arg("device") = 0, py::return_value_policy::copy);
}