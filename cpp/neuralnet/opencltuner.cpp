#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/openclhelpers.h"
#include "../neuralnet/opencltuner.h"
#include "../neuralnet/openclkernels.h"
#include "../core/rand.h"

#include <cstring>

using namespace std;
using namespace OpenCLHelpers;

string OpenCLTuneParams::XGemmDirectParams::desc() const {
  string s;
  s += "WGD=" + Global::intToString(WGD);
  s += " MDIMCD=" + Global::intToString(MDIMCD);
  s += " NDIMCD=" + Global::intToString(NDIMCD);
  s += " MDIMAD=" + Global::intToString(MDIMAD);
  s += " NDIMBD=" + Global::intToString(NDIMBD);
  s += " KWID=" + Global::intToString(KWID);
  s += " VWMD=" + Global::intToString(VWMD);
  s += " VWND=" + Global::intToString(VWND);
  s += " PADA=" + Global::intToString(PADA);
  s += " PADB=" + Global::intToString(PADB);
  return s;
}
string OpenCLTuneParams::XGemmDirectParams::compileOptions() const {
  string s;
  s += "-DWGD=" + Global::intToString(WGD);
  s += " -DMDIMCD=" + Global::intToString(MDIMCD);
  s += " -DNDIMCD=" + Global::intToString(NDIMCD);
  s += " -DMDIMAD=" + Global::intToString(MDIMAD);
  s += " -DNDIMBD=" + Global::intToString(NDIMBD);
  s += " -DKWID=" + Global::intToString(KWID);
  s += " -DVWMD=" + Global::intToString(VWMD);
  s += " -DVWND=" + Global::intToString(VWND);
  s += " -DPADA=" + Global::intToString(PADA);
  s += " -DPADB=" + Global::intToString(PADB);
  return s;
}

string OpenCLTuneParams::Conv3x3Params::desc() const {
  string s;
  s += "INTILE_XSIZE=" + Global::intToString(winograd_3x3_INTILE_XSIZE);
  s += " INTILE_YSIZE=" + Global::intToString(winograd_3x3_INTILE_YSIZE);
  s += " OUTTILE_XSIZE=" + Global::intToString(winograd_3x3_OUTTILE_XSIZE);
  s += " OUTTILE_YSIZE=" + Global::intToString(winograd_3x3_OUTTILE_YSIZE);
  return s;
}
string OpenCLTuneParams::Conv3x3Params::compileOptions() const {
  string s;
  s += "-DINTILE_XSIZE=" + Global::intToString(winograd_3x3_INTILE_XSIZE);
  s += " -DINTILE_YSIZE=" + Global::intToString(winograd_3x3_INTILE_YSIZE);
  s += " -DOUTTILE_XSIZE=" + Global::intToString(winograd_3x3_OUTTILE_XSIZE);
  s += " -DOUTTILE_YSIZE=" + Global::intToString(winograd_3x3_OUTTILE_YSIZE);
  return s;
}


bool OpenCLTuneParams::operator==(const OpenCLTuneParams& other) const {
  if(this == &other)
    return true;
  return std::memcmp(this,&other,sizeof(OpenCLTuneParams)) == 0;
}


static cl_mem randomReadOnlyBuffer(const char* seed, cl_context context, int numFloats, double scale) {
  vector<float> buf(numFloats);
  Rand rand(seed);
  for(int i = 0; i<numFloats; i++)
    buf[i] = rand.nextDouble(scale);
  return createReadOnlyBuffer(context,buf);
}


template<typename T>
static void addConfigs(
  vector<OpenCLTuneParams>& configs,
  std::function<void(OpenCLTuneParams&, T value)> apply,
  const vector<T>& values
) {
  vector<OpenCLTuneParams> newCfgs;
  for(int i = 0; i<values.size(); i++) {
    for(int j = 0; j<configs.size(); j++) {
      OpenCLTuneParams cfg = configs[j];
      apply(cfg,values[i]);
      newCfgs.push_back(cfg);
    }
  }
  configs = newCfgs;
}

#define SETTER(field) std::function<void(OpenCLTuneParams&, int value)>([](OpenCLTuneParams& p, int value){ p.field = value; })

OpenCLTuneParams OpenCLTuner::tune(
  const OpenCLTuneParams& initialConfig,
  int gpuIdx,
  Logger* logger,
  const int batchSize,
  const int nnXLen,
  const int nnYLen,
  const ModelDesc* model
) {
  bool enableProfiling = true;
  DevicesContext devicesContext({gpuIdx}, logger, enableProfiling);
  const cl_context& context = devicesContext.context;
  const std::vector<cl_device_id>& deviceIdsToUse = devicesContext.deviceIdsToUse;

  OpenCLTuneParams currentConfig = initialConfig;

  //Tune xGemmDirect
  {
    cout << "Tuning xGemmDirect for convolutions" << endl;

    vector<OpenCLTuneParams> configs;
    configs.push_back(initialConfig);
    addConfigs(configs,SETTER(xGemmDirect.WGD),{8,16,32,64});
    addConfigs(configs,SETTER(xGemmDirect.MDIMCD),{8,16,32});
    addConfigs(configs,SETTER(xGemmDirect.NDIMCD),{8,16,32});
    addConfigs(configs,SETTER(xGemmDirect.MDIMAD),{8,16,32});
    addConfigs(configs,SETTER(xGemmDirect.NDIMBD),{8,16,32});
    addConfigs(configs,SETTER(xGemmDirect.KWID),{2,8});
    addConfigs(configs,SETTER(xGemmDirect.VWMD),{1,2,4,8});
    addConfigs(configs,SETTER(xGemmDirect.VWND),{1,2,4,8});
    // addConfigs(configs,SETTER(xGemmDirect.PADA),{1,0});
    // addConfigs(configs,SETTER(xGemmDirect.PADB),{1,0});

    Rand rand;
    for(int i = configs.size()-1; i > 0; i--) {
      int j = rand.nextUInt(i+1);
      std::swap(configs[i],configs[j]);
    }
    //Make sure initial params is at the front
    bool foundIntital = false;
    for(int i = 0; i<configs.size(); i++) {
      if(configs[i] == initialConfig) {
        foundIntital = true;
        std::swap(configs[0],configs[i]);
      }
    }
    if(!foundIntital) {
      configs.push_back(initialConfig);
      std::swap(configs[0],configs[configs.size()-1]);
    }

    auto test = [&](OpenCLTuneParams& cfg, double& ret) {
      cl_int err;
      cl_program program = compileProgram("xgemmDirectProgram", context, deviceIdsToUse, OpenCLKernels::xgemmDirect, cfg.xGemmDirect.compileOptions());
      cl_kernel kernel = clCreateKernel(program, "XgemmDirectBatchedTT", &err);
      CHECK_ERR(err);

      int numTilesX = (nnXLen + cfg.conv3x3.winograd_3x3_OUTTILE_XSIZE - 1) / cfg.conv3x3.winograd_3x3_OUTTILE_XSIZE;
      int numTilesY = (nnYLen + cfg.conv3x3.winograd_3x3_OUTTILE_YSIZE - 1) / cfg.conv3x3.winograd_3x3_OUTTILE_YSIZE;
      int numTilesTotal = batchSize * numTilesX * numTilesY;

      int inTileXSize = cfg.conv3x3.winograd_3x3_INTILE_XSIZE;
      int inTileYSize = cfg.conv3x3.winograd_3x3_INTILE_YSIZE;
      int inTileXYSize = inTileXSize * inTileYSize;

      int maxChannels = std::max(
        std::max(model->trunk.trunkNumChannels, model->trunk.midNumChannels),
        std::max(model->trunk.regularNumChannels, model->trunk.gpoolNumChannels)
      );
      int ioNumFloats = numTilesTotal * maxChannels * inTileXYSize;
      int filterNumFloats = maxChannels * maxChannels * inTileXYSize;
      cl_mem input = randomReadOnlyBuffer("tune3x3Input", context, ioNumFloats, 1.0);
      cl_mem filter = randomReadOnlyBuffer("tune3x3Filter", context, filterNumFloats, 1.0 / sqrt(maxChannels * 3 * 3));
      cl_mem output = createReadWriteBuffer(context, ioNumFloats);

      bool bad = false;
      int numKernelsCounted = 0;
      double timeTaken = 0;

      const int reps = 9;
      for(int i = 0; i<reps; i++) {
        int inChannels = model->trunk.trunkNumChannels;
        int outChannels =
          i % 8 == 6 ? model->trunk.regularNumChannels :
          i % 8 == 7 ? model->trunk.gpoolNumChannels :
          model->trunk.midNumChannels;
        if(i % 8 == 1 || i % 8 == 3 || i % 8 == 5)
          std::swap(inChannels, outChannels);

        cl_event event;
        err = doBatchedXGemm_KM_KN_MN(
          kernel,
          devicesContext.commandQueues[0],
          cfg,
          outChannels, numTilesTotal, inChannels,
          filter, input, output,
          inTileXYSize,
          &event
        );
        if(err != 0) {
          bad = true;
          break;
        }

        err = clWaitForEvents(1, &event);
        CHECK_ERR(err);

        cl_ulong time_start, time_end;
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL); CHECK_ERR(err);
        err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL); CHECK_ERR(err);

        //Skip first kernel call to warm up
        if(i > 0) {
          timeTaken += (time_end - time_start) * 1e-9;
          numKernelsCounted++;
        }

        clReleaseEvent(event);
      }

      clReleaseMemObject(input);
      clReleaseMemObject(filter);
      clReleaseMemObject(output);

      clReleaseKernel(kernel);
      clReleaseProgram(program);

      if(bad)
        return false;

      double kernelsPerSecond = numKernelsCounted / timeTaken;
      ret = kernelsPerSecond;
      return true;
    };

    double bestKernelsPerSecond = 0.0;
    int numTested = 0;
    int numTestedRunnable = 0;
    for(int i = 0; i<configs.size(); i++) {
      double kernelsPerSecond;
      bool suc = test(configs[i],kernelsPerSecond);
      numTested++;
      if(suc) {
        numTestedRunnable++;
        if(kernelsPerSecond > bestKernelsPerSecond) {
          bestKernelsPerSecond = kernelsPerSecond;
          currentConfig = configs[i];
          cout << "Tuning " << i << "/"  << configs.size() << " Calls/sec " << bestKernelsPerSecond << " " << currentConfig.xGemmDirect.desc() << endl;
        }
      }
    }

  }

  return currentConfig;
}

#endif
