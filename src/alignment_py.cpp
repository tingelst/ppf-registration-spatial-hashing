#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
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
typename pcl::PointCloud<Point>::Ptr loadPLYFile(const std::string &file_name) {
  typename pcl::PointCloud<Point>::Ptr cloud(new pcl::PointCloud<Point>);
  pcl::io::loadPLYFile<Point>(file_name, *cloud);
  return cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr
randomDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float p) {
  pcl::RandomSample<Point> rs;
  typename pcl::PointCloud<Point>::Ptr filtered_cloud(
      new pcl::PointCloud<Point>);
  rs.setSample(p * cloud->size());
  rs.setInputCloud(cloud);
  rs.filter(*filtered_cloud);
  return filtered_cloud;
}

PYBIND11_DECLARE_HOLDER_TYPE(T, boost::shared_ptr<T>);

PYBIND11_MODULE(ppf_registration_spatial_hashing, m) {
  py::class_<pcl::PointNormal>(m, "PointNormal");

  auto point_cloud =
      py::class_<pcl::PointCloud<pcl::PointNormal>,
                 boost::shared_ptr<pcl::PointCloud<pcl::PointNormal>>>(
          m, "PointCloud")
          .def_readonly("width", &pcl::PointCloud<pcl::PointNormal>::width)
          .def_readonly("height", &pcl::PointCloud<pcl::PointNormal>::height)
          .def_readonly("points", &pcl::PointCloud<pcl::PointNormal>::points);

  m.def("load_ply_file", &loadPLYFile<pcl::PointNormal>);
  m.def("random_downsample", &randomDownsample<pcl::PointNormal>);
}
