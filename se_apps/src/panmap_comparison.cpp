#include <csignal>
#include <cstring>
#include <default_parameters.h>
#include <interface.h>
#include <se/DenseSLAMSystem.h>
#include <se/continuous/volume_template.hpp>
#include <sstream>
#include <stdint.h>
#include <string>
#include <time.h>
#include <vector>

#include <getopt.h>
#include <iomanip>
#include <perfstats.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <pcl/io/ply_io.h>

#ifndef __QT__
#include <draw.h>
#endif

PerfStats Stats;

// Params to just run this for evaluation purposes. The default values will be
// taken for everything.
struct Params {
  // Paths.
  const std::string data_path =
      "/home/lukas/Documents/Datasets/flat_dataset/run1/"; // With trailing
                                                           // slash.
  const std::string output_path =
      "/home/lukas/Documents/PanopticMapping/supereight/"; // With trailing
                                                           // slash.
  const std::string input_file =
      "/home/lukas/Documents/PanopticMapping/supereight/inputs/"
      "flat_run1.raw"; // Output of the processed flat sequence.
  const std::string ground_truth_pointcloud_file =
      "/home/lukas/Documents/Datasets/flat_dataset/ground_truth/run1/"
      "flat_1_gt_10000_visible.ply";

  // Environment params.
  const int res = 512;
  const Eigen::Vector3i volume_resolution = {res, res, res};
  const Eigen::Vector3f volume_size = {20, 20, 20};
  const Eigen::Vector3f initial_position = {10, 10, 10};
  const Eigen::Vector4f camera_K = {320, 320, 320, 240};

  // Supereight system params.
  const float truncation_distance = .3;

  // Evaluation params.
  const bool recompute_map = false;
  const bool visualize = true;
  const bool use_tracking = false;
  const float maximum_error_distance = 0.1; // m
  const bool count_truncated_points = false;
};

/***
 * This program loop over a scene recording.
 */
int main(int argc, char **argv) {

  Configuration config = parseArgs(argc, argv);
  Params params;

  // ========= CHECK ARGS =====================

  std::ostream *logstream = &std::cout;
  std::ofstream logfilestream;
  assert(config.compute_size_ratio > 0);
  assert(config.integration_rate > 0);
  assert(config.volume_size.x() > 0);
  assert(config.volume_resolution.x() > 0);
  logfilestream.open((params.output_path + "log.csv").c_str());
  logstream = &logfilestream;

  // Overwrite certain Config params.
  config.input_file = params.input_file;
  config.compute_size_ratio = 1;
  config.integration_rate = 1;
  config.camera = params.camera_K;
  config.volume_resolution = params.volume_resolution;
  config.volume_size = params.volume_size;

  // ========= READER INITIALIZATION  =========

  DepthReader *reader;

  if (is_file(config.input_file)) {
    reader =
        new RawDepthReader(config.input_file, config.fps, config.blocking_read);

  } else {
    std::cerr << "Input file '" << config.input_file << "' not found."
              << std::endl;
    return -1;
  }

  std::cout.precision(10);
  std::cerr.precision(10);

  Eigen::Vector3f init_pose = params.initial_position;
  const uint2 inputSize = reader->getinputSize();
  std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y
            << std::endl;

  //  =========  BASIC PARAMETERS  (input size / computation size )  =========

  const uint2 computationSize =
      make_uint2(inputSize.x / config.compute_size_ratio,
                 inputSize.y / config.compute_size_ratio);
  Eigen::Vector4f camera = config.camera;
  //  =========  BASIC BUFFERS  (input / output )  =========

  // Construction Scene reader and input buffer
  uint16_t *inputDepth =
      (uint16_t *)malloc(sizeof(uint16_t) * inputSize.x * inputSize.y);
  uchar3 *inputRGB =
      (uchar3 *)malloc(sizeof(uchar3) * inputSize.x * inputSize.y);
  uchar4 *depthRender =
      (uchar4 *)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);
  uchar4 *trackRender =
      (uchar4 *)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);
  uchar4 *volumeRender =
      (uchar4 *)malloc(sizeof(uchar4) * computationSize.x * computationSize.y);

  uint frame = 0;

  DenseSLAMSystem pipeline(
      Eigen::Vector2i(computationSize.x, computationSize.y),
      config.volume_resolution, config.volume_size, init_pose, config.pyramid,
      config);
  std::shared_ptr<se::Octree<FieldType>> map_ptr;

  if (params.recompute_map) {
    std::chrono::time_point<std::chrono::steady_clock> timings[7];
    timings[0] = std::chrono::steady_clock::now();

    *logstream
        << "frame\tacquisition\tpreprocessing\ttracking\tintegration\trayc"
           "asting\trendering\tcomputation\ttotal"
        << std::endl;
    logstream->setf(std::ios::fixed, std::ios::floatfield);

    while (reader->readNextDepthFrame(inputRGB, inputDepth)) {
      timings[1] = std::chrono::steady_clock::now();

      pipeline.preprocessing(inputDepth,
                             Eigen::Vector2i(inputSize.x, inputSize.y),
                             config.bilateralFilter);

      timings[2] = std::chrono::steady_clock::now();

      // get Pose.
      bool tracked = false;
      if (params.use_tracking) {
        tracked = pipeline.tracking(camera, config.icp_threshold,
                                    config.tracking_rate, frame);
      } else {
        // Read GT Pose
        Eigen::Matrix4f pose;
        std::stringstream pose_file_name;
        pose_file_name << params.data_path << std::setw(6) << std::setfill('0')
                       << frame << "_pose.txt";
        std::ifstream pose_file(pose_file_name.str());
        if (!pose_file.is_open()) {
          std::cerr << "Could not read file '" << pose_file_name.str() << "'."
                    << std::endl;
          return -1;
        }
        for (unsigned int i = 0; i < 4; i++) {
          for (unsigned int j = 0; j < 4; j++) {

            pose_file >> pose(i, j);
          }
        }
        pipeline.setPose(pose);
      }

      timings[3] = std::chrono::steady_clock::now();

      // Integrate
      bool integrated = pipeline.integration(camera, config.integration_rate,
                                             config.mu, frame);

      timings[4] = std::chrono::steady_clock::now();

      pipeline.raycasting(camera, config.mu, frame);

      timings[5] = std::chrono::steady_clock::now();

      // Draw
      if (params.visualize) {
        pipeline.renderDepth(
            (unsigned char *)depthRender,
            Eigen::Vector2i(computationSize.x, computationSize.y));
        pipeline.renderTrack(
            (unsigned char *)trackRender,
            Eigen::Vector2i(computationSize.x, computationSize.y));
        pipeline.renderVolume(
            (unsigned char *)volumeRender,
            Eigen::Vector2i(computationSize.x, computationSize.y), frame,
            config.rendering_rate, camera, 0.75 * config.mu);

        drawthem(inputRGB, depthRender, trackRender, volumeRender, trackRender,
                 inputSize, computationSize, computationSize, computationSize);
      }
      timings[6] = std::chrono::steady_clock::now();

      *logstream
          << frame << "\t"
          << std::chrono::duration<double>(timings[1] - timings[0]).count()
          << "\t" //  acquisition
          << std::chrono::duration<double>(timings[2] - timings[1]).count()
          << "\t" //  preprocessing
          << std::chrono::duration<double>(timings[3] - timings[2]).count()
          << "\t" //  tracking
          << std::chrono::duration<double>(timings[4] - timings[3]).count()
          << "\t" //  integration
          << std::chrono::duration<double>(timings[5] - timings[4]).count()
          << "\t" //  raycasting
          << std::chrono::duration<double>(timings[6] - timings[5]).count()
          << "\t" //  rendering
          << std::chrono::duration<double>(timings[5] - timings[1]).count()
          << "\t" //  computation
          << std::chrono::duration<double>(timings[6] - timings[0])
                 .count() //  total
          << std::endl;

      std::cout << "processed frame " << frame << "." << std::endl;
      frame++;
      timings[0] = std::chrono::steady_clock::now();
    }
    std::cout << "saving map." << std::endl;
    pipeline.getMap(map_ptr);
    map_ptr->save(params.output_path + "map.bin");
    auto mesh = pipeline.get_mesh();
    pipeline.dump_mesh(params.output_path + "mesh.vtk", mesh);
  } else {
    std::cout << "Loaded map." << std::endl;
    map_ptr = std::make_shared<se::Octree<FieldType>>();
    map_ptr->load(params.output_path + "map.bin");
  }

  // ========== Evaluate error ==========

  std::cout << "Computing reconstruction error." << std::endl;
  auto gt_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(params.ground_truth_pointcloud_file,
                                          *gt_ptcloud_) != 0) {
    std::cerr << "Could not load ground truth point cloud from '"
              << params.ground_truth_pointcloud_file << "'." << std::endl;
    return -1;
  }

  auto volume = VolumeTemplate<FieldType, se::Octree>(
      params.volume_resolution.x(), params.volume_size.x(), map_ptr.get());

  auto select_sdf =
      [](const typename se::Octree<FieldType>::value_type &value) {
        return value.x;
      };
  auto select_weight =
      [](const typename se::Octree<FieldType>::value_type &value) {
        return value.y;
      };
  const float inverseVoxelSize =
      params.volume_resolution.x() / params.volume_size.x();

  std::ofstream output_file((params.output_path + "evaluation.csv").c_str());
  output_file << "MeanError [m],StdError [m],RMSE [m],TotalPoints [1],"
              << "UnknownPoints [1],TruncatedPoints [1]\n";

  for (int scale = 0; scale < max_scale; ++scale) {
    std::cout << "Scale: " << scale << std::endl;
    uint64_t total_points = 0;
    uint64_t unknown_points = 0;
    uint64_t truncated_points = 0;
    std::vector<float> abserror;

    for (const auto &point : *gt_ptcloud_) {
      Eigen::Vector3f position =
          Eigen::Vector3f(point.x, point.y, point.z) + params.initial_position;

      Eigen::Vector3f discrete_pos = (inverseVoxelSize * position);
      // float distance = map_ptr->interp(discrete_pos, i, select_sdf).first *
      //                  params.truncation_distance;
      // float weight = map_ptr->interp(discrete_pos, select_weight).first *
      //                params.truncation_distance;

      float distance = volume.interp(position, scale, select_sdf).first;
      float weight = volume.interp(position, scale, select_weight).first;

      if (weight > 1e-6) {
        if (std::abs(distance) > params.maximum_error_distance) {
          truncated_points++;
          if (!params.count_truncated_points) {
            continue;
          }
          distance = params.maximum_error_distance;
        }
        abserror.push_back(std::abs(distance));
      } else {
        unknown_points++;
      }
    }
    // Report summary.
    float mean = 0.0;
    float rmse = 0.0;
    for (auto value : abserror) {
      mean += value;
      rmse += std::pow(value, 2);
    }
    if (!abserror.empty()) {
      mean /= static_cast<float>(abserror.size());
      rmse = std::sqrt(rmse / static_cast<float>(abserror.size()));
    }
    float stddev = 0.0;
    for (auto value : abserror) {
      stddev += std::pow(value - mean, 2.0);
    }
    if (abserror.size() > 2) {
      stddev = sqrt(stddev / static_cast<float>(abserror.size() - 1));
    }
    output_file << mean << "," << stddev << "," << rmse << ","
                << gt_ptcloud_->size() << "," << unknown_points << ","
                << truncated_points << "\n";
  }

  std::cout << "Done." << std::endl;

  //  =========  FREE BASIC BUFFERS  =========

  free(inputDepth);
  free(depthRender);
  free(trackRender);
  free(volumeRender);
  return 0;
}
