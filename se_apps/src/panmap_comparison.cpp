#include <csignal>
#include <cstring>
#include <default_parameters.h>
#include <interface.h>
#include <se/DenseSLAMSystem.h>
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
      "/home/lukas/Documents/PanopticMapping/supereight/"
      "flat_run1.raw"; // Output of the processed flat sequence.
  const std::string ground_truth_pointcloud_file =
      "/home/lukas/Documents/Datasets/flat_dataset/ground_truth/run1/"
      "flat_1_gt_10000.ply";

  // Environment params.
  const Eigen::Vector3f volume_size = {20, 20, 4};
  const Eigen::Vector3i volume_resolution = {256, 256, 256};
  const Eigen::Vector3f initial_position = {10, 10, 2}; //{10.f, 10.f, 2.f};
  const Eigen::Vector4f camera_K = {320, 320, 320, 240};

  // Supereight system params.
  const float truncation_distance = .3;

  // Evaluation params.
  const bool recompute_map = true;
  const float maximum_error_distance = 0.2; // m
};

float selectSDF(const typename se::Octree<FieldType>::value_type &value) {
  return value.x;
}

/***
 * This program loop over a scene recording
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
  Eigen::Vector4f camera = reader->getK() / config.compute_size_ratio;

  if (config.camera_overrided)
    camera = config.camera / config.compute_size_ratio;
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
           "asting\trendering\tcomputation\ttotal    \tX          \tY     "
           "     \tZ         \ttracked   \tintegrated"
        << std::endl;
    logstream->setf(std::ios::fixed, std::ios::floatfield);

    while (reader->readNextDepthFrame(inputRGB, inputDepth)) {

      timings[1] = std::chrono::steady_clock::now();

      pipeline.preprocessing(inputDepth,
                             Eigen::Vector2i(inputSize.x, inputSize.y),
                             config.bilateralFilter);

      timings[2] = std::chrono::steady_clock::now();

      // get Pose.
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
      Eigen::Matrix4f transform;
      transform << 0.0000000f, -1.0000000f, 0.0000000f, 0.0f, 0.0000000f,
          0.0000000f, -1.0000000f, 0.0f, 1.0000000f, 0.0000000f, 0.0000000f,
          0.0f, 0.0f, 0.0f, 0.0f, 1.0f;
      // pose = transform * pose;
      pose(0, 3) += params.initial_position.x();
      pose(1, 3) += params.initial_position.y();
      pose(2, 3) += params.initial_position.z();

      pose.setIdentity();
      pipeline.setPose(pose);
      bool tracked = pipeline.tracking(camera, config.icp_threshold,
                                       config.tracking_rate, frame);
                                             pipeline.setPose(pose);

      std::cout << "Tracked: " << tracked << std::endl;

      float xt = pose(0, 3) - init_pose.x();
      float yt = pose(1, 3) - init_pose.y();
      float zt = pose(2, 3) - init_pose.z();

      timings[3] = std::chrono::steady_clock::now();

      // Integrate
      bool integrated = pipeline.integration(camera, config.integration_rate,
                                             config.mu, frame);
      std::cout << "Integrated: " << integrated << std::endl;

      timings[4] = std::chrono::steady_clock::now();

      pipeline.raycasting(camera, config.mu, frame);

      timings[5] = std::chrono::steady_clock::now();

      // Draw
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
          << std::chrono::duration<double>(timings[6] - timings[0]).count()
          << "\t" //  total
          << xt << "\t" << yt << "\t" << zt
          << "\t" //  X,Y,Z
                  //  << tracked << "        \t"
                  //  << integrated // tracked and integrated flags
          << std::endl;

      std::cout << "processed frame " << frame << "." << std::endl;
      frame++;
      timings[0] = std::chrono::steady_clock::now();
      if (frame > 5)
        break;
    }
    std::cout << "saving map." << std::endl;
    pipeline.getMap(map_ptr);
    map_ptr->save(params.output_path + "map.bin");
    pipeline.dump_mesh(params.output_path + "mesh.vtk");
  } else {
    std::cout << "Loaded map." << std::endl;
    map_ptr = std::make_shared<se::Octree<FieldType>>();
    map_ptr->load(params.output_path + "map.bin");
  }

  // ========== Evaluate error ==========

  std::cout << "Computing reconstruction error." << std::endl;

  uint64_t total_points = 0;
  uint64_t unknown_points = 0;
  uint64_t truncated_points = 0;
  std::vector<float> abserror;
  auto gt_ptcloud_ = std::make_unique<pcl::PointCloud<pcl::PointXYZ>>();
  if (pcl::io::loadPLYFile<pcl::PointXYZ>(params.ground_truth_pointcloud_file,
                                          *gt_ptcloud_) != 0) {
    std::cerr << "Could not load ground truth point cloud from '"
              << params.ground_truth_pointcloud_file << "'." << std::endl;
    return -1;
  }

  for (const auto &point : *gt_ptcloud_) {
    Eigen::Vector3f position =
        Eigen::Vector3f(point.x, point.y, point.z) - params.initial_position;
    float distance =
        map_ptr->interp(position, selectSDF).first * params.truncation_distance;
    // std::cout << position.transpose() << ": " << distance << std::endl;
    if (distance < params.truncation_distance &&
        distance > -params.truncation_distance) {
      if (std::abs(distance) > params.maximum_error_distance) {
        distance = params.maximum_error_distance;
        truncated_points++;
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
  std::ofstream output_file((params.output_path + "evaluation.csv").c_str());
  output_file << "MeanError [m],StdError [m],RMSE [m],TotalPoints [1],"
              << "UnknownPoints [1],TruncatedPoints [1]\n";
  output_file << mean << "," << stddev << "," << rmse << "," << total_points
              << "," << unknown_points << "," << truncated_points << "\n";

  std::cout << "Done." << std::endl;

  // ==========     DUMP VOLUME      =========

  // if (config.dump_volume_file != "") {
  //   auto s = std::chrono::steady_clock::now();
  //   pipeline.dump_volume(config.dump_volume_file);
  //   auto e = std::chrono::steady_clock::now();
  //   std::cout << "Mesh generated in " << (e - s).count() << " seconds"
  //             << std::endl;
  // }

  //  =========  FREE BASIC BUFFERS  =========

  free(inputDepth);
  free(depthRender);
  free(trackRender);
  free(volumeRender);
  return 0;
}
