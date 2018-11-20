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

// Workaround for a bug in pcl::geometry::distance.
template <typename PointT>
inline float distance(const PointT &p1, const PointT &p2)
{
    Eigen::Vector3f diff = p1.getVector3fMap() - p2.getVector3fMap();
    return (diff.norm());
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr
randomDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float p)
{
    pcl::RandomSample<Point> rs;
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(
        new pcl::PointCloud<Point>);
    rs.setSample(p * cloud->size());
    rs.setInputCloud(cloud);
    rs.filter(*filtered_cloud);
    return filtered_cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr
sequentialDownsample(typename pcl::PointCloud<Point>::Ptr cloud, int n)
{
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(
        new pcl::PointCloud<Point>);
    int i = 0;
    for (typename pcl::PointCloud<Point>::iterator it = cloud->begin();
         it != cloud->end(); ++it, ++i)
    {
        if (i % n == 0)
        {
            filtered_cloud->push_back(*it);
        }
    }
    return filtered_cloud;
}

template <typename Point>
typename pcl::PointCloud<Point>::Ptr
voxelGridDownsample(typename pcl::PointCloud<Point>::Ptr cloud, float leaf)
{
    typename pcl::PointCloud<Point>::Ptr filtered_cloud(
        new pcl::PointCloud<Point>);
    pcl::VoxelGrid<Point> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(leaf, leaf, leaf);
    sor.filter(*filtered_cloud);
    return filtered_cloud;
}

/*

int run(int argc, char **argv)
{
    srand(time(0));
    po::variables_map vm = configure_options(argc, argv);
    init_logging(vm);

    std::string argv_string;
    for (int i = 0; i < argc; i++)
    {
        argv_string.append(argv[i]);
        argv_string.append(std::string(" "));
    }
    BOOST_LOG_TRIVIAL(info) << argv_string;

    float scene_leaf_size = vm["scene_leaf_size"].as<float>();
    std::vector<float> scene_d_dists;
    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> scene_clouds;
    CommaSeparatedVector scene_files =
        vm["scene_files"].as<CommaSeparatedVector>();
    for (std::string scene_file : scene_files.values)
    {
        scene_clouds.push_back(pcl::PointCloud<pcl::PointNormal>::Ptr(
            new pcl::PointCloud<pcl::PointNormal>));
        BOOST_LOG_TRIVIAL(info)
            << boost::format("Loading scene point cloud: %s") %
scene_file.c_str(); if (pcl::io::loadPLYFile<pcl::PointNormal>(scene_file,
                                                   *scene_clouds.back()) < 0)
        {
            BOOST_LOG_TRIVIAL(error) << "Error loading scene file!";
            exit(1);
        }

        scene_d_dists.push_back(scene_leaf_size);
    }

    std::vector<float> tau_d;
    CommaSeparatedVector tau_d_strs = vm["tau_d"].as<CommaSeparatedVector>();
    for (std::string tau_d_str : tau_d_strs.values)
    {
        tau_d.push_back(std::atof(tau_d_str.c_str()));
    }

    std::vector<pcl::PointCloud<pcl::PointNormal>::Ptr> model_clouds;
    std::vector<float> model_d_dists;
    CommaSeparatedVector model_files =
        vm["model_files"].as<CommaSeparatedVector>();

    if (tau_d.size() != model_files.values.size())
    {
        BOOST_LOG_TRIVIAL(error) << "Each model must have an associated tau_d.";
        exit(1);
    }
    for (int i = 0; i < model_files.values.size(); i++)
    {
        std::string model_file = model_files.values[i];
        model_clouds.push_back(pcl::PointCloud<pcl::PointNormal>::Ptr(
            new pcl::PointCloud<pcl::PointNormal>));

        BOOST_LOG_TRIVIAL(info)
            << boost::format("Loading model point cloud: %s") %
model_file.c_str(); if (pcl::io::loadPLYFile<pcl::PointNormal>(model_file,
                                                   *model_clouds.back()) < 0)
        {
            BOOST_LOG_TRIVIAL(error) << "Error loading model file!";
            exit(1);
        }

        pcl::PointNormal minPt, maxPt;
        // Finding the max pairwise distance is epxensive, so
        // approximate it with the max difference between coords.
        pcl::getMinMax3D(*model_clouds.back(), minPt, maxPt);
        float3 model_diam = {maxPt.x - minPt.x, maxPt.y - minPt.y,
                             maxPt.z - minPt.z};
        model_d_dists.push_back(tau_d[i] * max(model_diam));
        BOOST_LOG_TRIVIAL(debug)
            << boost::format("model_diam, d_dist: (%f, %f, %f), %f") %
                   model_diam.x % model_diam.y % model_diam.z %
                   model_d_dists.back();
    }

    // Downsample
    BOOST_LOG_TRIVIAL(info) << "Downsampling...";

    // shared_ptr &operator=(shared_ptr const &r) is equivalent to
    // shared_ptr(r).swap(*this), so, in each loop, the original cloud gets
    // de-allocated when new_* goes out of scope.
    //
http://www.boost.org/doc/libs/1_60_0/libs/smart_ptr/shared_ptr.htm#assignment
    for (pcl::PointCloud<pcl::PointNormal>::Ptr &scene : scene_clouds)
    {
        BOOST_LOG_TRIVIAL(info)
            << boost::format("Scene size before filtering: %u (%u x %u)") %
                   scene->size() % scene->width % scene->height;
        pcl::PointCloud<pcl::PointNormal>::Ptr new_scene =
            voxelGridDownsample<pcl::PointNormal>(scene, scene_leaf_size);
        scene = new_scene;
        BOOST_LOG_TRIVIAL(info)
            << boost::format("Scene size after filtering: %u (%u x %u)") %
                   scene->size() % scene->width % scene->height;
    }

    for (int i = 0; i < model_clouds.size(); i++)
    {
        pcl::PointCloud<pcl::PointNormal>::Ptr model = model_clouds[i];
        BOOST_LOG_TRIVIAL(info)
            << boost::format("Model size before filtering: %u (%u x %u)") %
                   model->size() % model->width % model->height;
        pcl::PointCloud<pcl::PointNormal>::Ptr new_model =
            voxelGridDownsample<pcl::PointNormal>(model, model_d_dists[i]);
        BOOST_LOG_TRIVIAL(info)
            << boost::format("Model size after filtering: %u (%u x %u)") %
                   new_model->size() % new_model->width % new_model->height;
        model_clouds[i] = new_model;
    }

    std::vector<std::vector<Eigen::Matrix4f>> results = ppf_registration(
        scene_clouds, model_clouds, model_d_dists,
        vm["ref_point_df"].as<unsigned int>(),
        vm["vote_count_threshold"].as<float>(), vm["cpu_clustering"].as<bool>(),
        vm["use_l1_norm"].as<bool>(), vm["use_averaged_clusters"].as<bool>(),
        vm["dev"].as<int>(), NULL);

    return 0;
}
*/