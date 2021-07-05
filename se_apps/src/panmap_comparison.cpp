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

#include <glog/logging.h>

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

  // Environment params.
  const Eigen::Vector3f volume_size = {2.f, 2.f, 2.f};
  const Eigen::Vector3i volume_resolution = {256, 256, 256};

  // Supereight system params.
	cosnt float truncation_distance = .5;
};

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
	  if (config.log_file != "") {
    logfilestream.open(config.log_file.c_str());
    logstream = &logfilestream;
  }

  // Overwrite certain Config params.
  config.input_file = params.input_file;
  config.compute_size_ratio = 1;
	config.integration_rate = 1;
	config.m


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

  Eigen::Vector3f init_pose = Eigen::Vector3f::Zero();
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

  std::chrono::time_point<std::chrono::steady_clock> timings[7];
  timings[0] = std::chrono::steady_clock::now();

  *logstream << "frame\tacquisition\tpreprocessing\ttracking\tintegration\trayc"
                "asting\trendering\tcomputation\ttotal    \tX          \tY     "
                "     \tZ         \ttracked   \tintegrated"
             << std::endl;
  logstream->setf(std::ios::fixed, std::ios::floatfield);

  while (reader->readNextDepthFrame(inputDepth)) {

    timings[1] = std::chrono::steady_clock::now();

    pipeline.preprocessing(inputDepth,
                           Eigen::Vector2i(inputSize.x, inputSize.y),
                           config.bilateralFilter);

    timings[2] = std::chrono::steady_clock::now();

    // tracked = pipeline.tracking(camera, config.icp_threshold,
    // 		config.tracking_rate, frame);

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
		pipeline.setPose(pose);

    float xt = pose(0, 3) - init_pose.x();
    float yt = pose(1, 3) - init_pose.y();
    float zt = pose(2, 3) - init_pose.z();

		    timings[3] = std::chrono::steady_clock::now();

    // Integrate 
    pipeline.integration(camera, config.integration_rate, config.mu, frame);

    timings[4] = std::chrono::steady_clock::now();

    pipeline.raycasting(camera, config.mu, frame);

    timings[5] = std::chrono::steady_clock::now();

    pipeline.renderDepth((unsigned char *)depthRender,
                         Eigen::Vector2i(computationSize.x, computationSize.y));
    pipeline.renderTrack((unsigned char *)trackRender,
                         Eigen::Vector2i(computationSize.x, computationSize.y));
    pipeline.renderVolume((unsigned char *)volumeRender,
                          Eigen::Vector2i(computationSize.x, computationSize.y),
                          frame, config.rendering_rate, camera,
                          0.75 * config.mu);

    timings[6] = std::chrono::steady_clock::now();

    *logstream << frame << "\t"
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

    frame++;
    timings[0] = std::chrono::steady_clock::now();
  }

  std::shared_ptr<se::Octree<FieldType>> map_ptr;
  pipeline.getMap(map_ptr);
  map_ptr->save("test.bin");

  // ==========     DUMP VOLUME      =========

  if (config.dump_volume_file != "") {
    auto s = std::chrono::steady_clock::now();
    pipeline.dump_volume(config.dump_volume_file);
    auto e = std::chrono::steady_clock::now();
    std::cout << "Mesh generated in " << (e - s).count() << " seconds"
              << std::endl;
  }

  //  =========  FREE BASIC BUFFERS  =========

  free(inputDepth);
  free(depthRender);
  free(trackRender);
  free(volumeRender);
  return 0;
}
