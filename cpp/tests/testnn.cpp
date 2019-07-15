#include "../tests/tests.h"
#include "../neuralnet/nninterface.h"

#include <cmath>

using namespace std;

static bool approxEqual(float x, float y, bool useFP16) {
  float tolerance;
  if(useFP16)
    tolerance = 0.01f * std::max(std::abs(x),std::max(std::abs(y),1.0f));
  else
    tolerance = 0.0001f * std::max(std::abs(x),std::max(std::abs(y),1.0f));
  return std::abs(x - y) < tolerance;
}

static void checkApproxEqual(
  const string& label,
  const vector<float>& vec, const vector<float>& expected, int nSize, int cSize, int ySize, int xSize, bool useFP16,
  const char* file, const char* func, int line
) {
  int cyxSize = cSize * ySize * xSize;
  int yxSize = ySize * xSize;

  bool mismatch = false;
  for(int n = 0; n < nSize; n++) {
    for(int c = 0; c < cSize; c++) {
      for(int y = 0; y < ySize; y++) {
        for(int x = 0; x < xSize; x++) {
          int i = n * cyxSize + c * yxSize + y * xSize + x;
          if(!approxEqual(vec[i],expected[i],useFP16) && !mismatch) {
            mismatch = true;
            cout << "File " << file << " func " << func << " line " << line << endl;
            cout << label << endl;
            cout << "Test failed at n c y x = " << n << " " << c << " " << y << " " << x << endl;
          }
        }
      }
    }
  }
  if(mismatch) {
    cout << "==========" << endl;
    cout << "Actual" << endl;
    cout << "==========" << endl;
    for(int n = 0; n < nSize; n++) {
      for(int c = 0; c < cSize; c++) {
        for(int y = 0; y < ySize; y++) {
          for(int x = 0; x < xSize; x++) {
            int i = n * cyxSize + c * yxSize + y * xSize + x;
            cout << Global::strprintf("%.5g, ",vec[i]);
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << "-------" << endl;
    }
    cout << "==========" << endl;
    cout << "Expected" << endl;
    cout << "==========" << endl;
    for(int n = 0; n < nSize; n++) {
      for(int c = 0; c < cSize; c++) {
        for(int y = 0; y < ySize; y++) {
          for(int x = 0; x < xSize; x++) {
            int i = n * cyxSize + c * yxSize + y * xSize + x;
            cout << Global::strprintf("%.5g, ",expected[i]);
          }
          cout << endl;
        }
        cout << endl;
      }
      cout << "-------" << endl;
    }
  }
}
#define CHECK_APPROX_EQUAL(label,vec,expected,n,c,h,w,useFP16) (checkApproxEqual((label),(vec),(expected),(n),(c),(h),(w),(useFP16),__FILE__,#vec,__LINE__))


static vector<float> NCHWtoNHWC(const vector<float>& vec, int nSize, int cSize, int ySize, int xSize) {
  vector<float> ret(vec.size());
  int cyxSize = cSize * ySize * xSize;
  int yxSize = ySize * xSize;
  int xcSize = xSize * cSize;
  for(int n = 0; n < nSize; n++) {
    for(int c = 0; c < cSize; c++) {
      for(int y = 0; y < ySize; y++) {
        for(int x = 0; x < xSize; x++) {
          ret[n * cyxSize + y * xcSize + x * cSize + c] = vec[n * cyxSize + c * yxSize + y * xSize + x];
        }
      }
    }
  }
  return ret;
}


static void testConvLayer(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const ConvLayerDesc& desc, const vector<float>& input, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.inChannels,nnYLen,nnXLen) : input;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.outChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateConv(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.outChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.outChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    int batchSize = 2;
    int inChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      5,5,4,4,
      5,5,4,4,
      1,1,8,8,

      0,1,2,3,
      3,4,5,6,
      8,7,6,5,

      0,1,0,2,
      3,0,4,0,
      0,5,0,6,

      1,0,0,2,
      0,2,2,0,
      0,2,2,0,
    });

    {
      string label("1x1 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          0,1,
          1,-1,
          10,0.1,
      });
      //NCHW
      vector<float> expected({
        0.0f, 1.0f, 2.0f, 3.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        8.0f, 7.0f, 6.0f, 5.0f,

        5.0f, 4.0f, 2.0f, 1.0f,
        2.0f, 1.0f, -1.0f, -2.0f,
        -7.0f, -6.0f, 2.0f, 3.0f,

        50.0f, 50.1f, 40.2f, 40.3f,
        50.3f, 50.4f, 40.5f, 40.6f,
        10.8f, 10.7f, 80.6f, 80.5f,

        1.0f, 0.0f, 0.0f, 2.0f,
        0.0f, 2.0f, 2.0f, 0.0f,
        0.0f, 2.0f, 2.0f, 0.0f,

        -1.0f, 1.0f, 0.0f, 0.0f,
        3.0f, -2.0f, 2.0f, 0.0f,
        0.0f, 3.0f, -2.0f, 6.0f,

        0.1f, 10.0f, 0.0f, 20.2f,
        30.0f, 0.2f, 40.2f, 0.0f,
        0.0f, 50.2f, 0.2f, 60.0f,
      });

      ConvLayerDesc desc;
      desc.convYSize = 1;
      desc.convXSize = 1;
      desc.inChannels = inChannels;
      desc.outChannels = 3;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

    {
      string label("3x3 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          1,0,0,
          0,0,0,
          0,0,0,

          0,0,0,
          0,0,0,
          0,0,0,

          0,0,0,
          0,0,1,
          0,1,0,

          0,0,0,
          0,-1,0,
          0,0,0,

          0,0,0,
          0,1,0,
          0,0,0,

          0,0,0,
          0,0,0,
          0,0,2,
      });
      //NCHW
      vector<float> expected({
        0, 0, 0, 0,
        0, 5, 5, 4,
        0, 5, 5, 4,

        10, 8, 6, 1,
        3, 1, 7, 2,
        -7, 1, 2, -5,

        13, 15, 16, 4,
        19, 17, 14, 4,
        1, 1, 8, 8,

        0, 0, 0, 0,
        0, 0, 1, 0,
        0, 3, 0, 4,

        3, 0, 6, -2,
        0, 7, -2, 6,
        5, -2, 4, 0,

        4, 5, 0, 2,
        7, 4, 4, 0,
        0, 5, 0, 6,
      });

      ConvLayerDesc desc;
      desc.convYSize = 3;
      desc.convXSize = 3;
      desc.inChannels = inChannels;
      desc.outChannels = 3;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

    {
      string label("5x5 convolution");

      //oc,ic,y,x
      vector<float> convWeights({
          0,0,0,0,1,
          0,0,0,1,0,
          0,0,1,0,0,
          0,0,0,0,0,
          0,0,0,0,0,

          0,0,0,0,0,
          0,0,0,0,0,
          0,0,1,0,0,
          0,1,0,0,0,
          1,0,0,0,0,

          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,2,

          0,0,0,0,0,
          0,0,1,0,0,
          2,0,0,0,0,
          0,0,0,0,0,
          0,0,0,0,0,
      });

      //NCHW
      vector<float> expected({
        5, 9,18,19,
       13,21,20,16,
       18,16,18,13,

       16,16, 0, 2,
        0, 1, 8,11,
        3, 4,21,20,

        1, 1, 2, 8,
        4, 2,10, 2,
        0,13, 2, 6,

        0,12, 2, 0,
        1, 0, 0, 6,
        0, 2, 2, 4,
      });

      ConvLayerDesc desc;
      desc.convYSize = 5;
      desc.convXSize = 5;
      desc.inChannels = inChannels;
      desc.outChannels = 2;
      desc.dilationY = 1;
      desc.dilationX = 1;
      desc.weights = convWeights;

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

  }


}


static void testBatchNormLayer(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const BatchNormLayerDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateBatchNorm(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    int batchSize = 2;
    int numChannels = 2;
    int nnYLen = 2;
    int nnXLen = 5;

    //NCHW
    vector<float> input({
        5,5,4,4,9,
        1,1,8,8,9,

        0,1,2,3,4,
        8,7,6,5,4,

        3,0,4,0,5,
        0,5,0,6,0,

        1,0,0,2,1,
        0,2,2,0,2,
    });

    {
      string label("Batch norm");

      BatchNormLayerDesc desc;
      desc.numChannels = numChannels;
      desc.epsilon = 0.1;
      desc.hasScale = true;
      desc.hasBias = true;
      desc.mean = vector<float>({0,2});
      desc.variance = vector<float>({3.9,0.15});
      desc.scale = vector<float>({0.1,1});
      desc.bias = vector<float>({10.0,0.0});

      vector<float> mask({
        1,1,1,1,1,
        1,1,1,1,1,

        1,1,1,1,1,
        1,1,1,1,1,
      });

      //NCHW
      vector<float> expected({
          10.25, 10.25, 10.2, 10.2, 10.45,
          10.05, 10.05, 10.4, 10.4, 10.45,

          -4, -2, 0, 2, 4,
          12, 10, 8, 6, 4,

          10.15, 10, 10.2, 10, 10.25,
          10, 10.25, 10, 10.3, 10,

          -2, -4, -4, 0, -2,
          -4, 0, 0, -4, 0,
      });
      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
    }

    {
      string label("Batch norm with mask");

      BatchNormLayerDesc desc;
      desc.numChannels = numChannels;
      desc.epsilon = 0.1;
      desc.hasScale = false;
      desc.hasBias = true;
      desc.mean = vector<float>({0,2});
      desc.variance = vector<float>({3.9,0.15});
      desc.scale = vector<float>({1,1});
      desc.bias = vector<float>({10.0,0.0});

      vector<float> mask({
        1,1,1,0,0,
        1,1,1,0,0,

        1,1,1,1,1,
        0,0,0,0,0,
      });

      //NCHW
      vector<float> expected({
          12.5, 12.5, 12, 0, 0,
          10.5, 10.5, 14, 0, 0,

          -4, -2, 0, 0, 0,
          12, 10, 8, 0, 0,

          11.5, 10, 12, 10, 12.5,
          0, 0, 0, 0, 0,

          -2, -4, -4, 0, -2,
          0, 0, 0, 0, 0,
      });

      testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
    }

  }

}


static void testResidualBlock(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const ResidualBlockDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateResidualBlock(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.preBN.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.preBN.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    string label("Basic residual block");

    int batchSize = 2;
    int trunkChannels = 1;
    int midChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      1,0,0,0,
      0,2,2,0,
      0,0,0,1,

      0,0,0,0,
      0,3,-5,0,
      1,1,1,1,
    });

    //Also, mask out some values
    vector<float> mask({
      1,1,0,1,
      1,1,1,1,
      1,1,0,1,

      1,1,1,1,
      1,1,1,0,
      1,1,1,1,
    });

    ResidualBlockDesc desc;

    //Doubles all values
    desc.preBN.name = "preBN";
    desc.preBN.numChannels = trunkChannels;
    desc.preBN.epsilon = 0.1;
    desc.preBN.hasScale = true;
    desc.preBN.hasBias = true;
    desc.preBN.mean = vector<float>({0});
    desc.preBN.variance = vector<float>({0.9});
    desc.preBN.scale = vector<float>({2});
    desc.preBN.bias = vector<float>({0.0});

    //ReLU gets applied, smooshing negatives
    //2,0,0,3,
    //0,4,4,0,
    //0,0,0,2,

    //0,0,0,0,
    //0,6,0,0,
    //2,2,2,2,

    //Split into two channels, shifting up and shifting down.
    desc.regularConv.name = "regularConv";
    desc.regularConv.convYSize = 3;
    desc.regularConv.convXSize = 3;
    desc.regularConv.inChannels = trunkChannels;
    desc.regularConv.outChannels = midChannels;
    desc.regularConv.dilationY = 1;
    desc.regularConv.dilationX = 1;
    desc.regularConv.weights = vector<float>({
        0,1,0,
        0,0,0,
        0,0,0,

        0,0,0,
        0,0,0,
        0,1,0,
    });
    //0,0,0,0,
    //2,0,0,3,
    //0,4,0,0,

    //0,4,0,0,
    //0,0,0,2,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,6,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,0,0,0,

    //Subtract 3 from all values in the 0th channel
    desc.midBN.name = "midBN";
    desc.midBN.numChannels = midChannels;
    desc.midBN.epsilon = 0.1;
    desc.midBN.hasScale = false;
    desc.midBN.hasBias = false;
    desc.midBN.mean = vector<float>({3,0});
    desc.midBN.variance = vector<float>({0.9,0.9});
    desc.midBN.scale = vector<float>({1,1});
    desc.midBN.bias = vector<float>({0.0,0.0});

    //ReLU gets applied, smooshing negatives
    //0,0,0,0,
    //0,0,0,0,
    //0,1,0,0,

    //0,4,0,0,
    //0,0,0,2,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,3,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,0,0,0,


    //Sum pointwise
    desc.finalConv.name = "finalConv";
    desc.finalConv.convYSize = 1;
    desc.finalConv.convXSize = 1;
    desc.finalConv.inChannels = midChannels;
    desc.finalConv.outChannels = trunkChannels;
    desc.finalConv.dilationY = 1;
    desc.finalConv.dilationX = 1;
    desc.finalConv.weights = vector<float>({
        1,1
    });

    //0,4,0,0,
    //0,0,0,2,
    //0,1,0,0,

    //0,6,0,0,
    //2,2,2,0,
    //0,3,0,0,

    //Then add to the original which was:

    //1,0,0,0,
    //0,2,2,0,
    //0,0,0,1,

    //0,0,0,0,
    //0,3,-5,0,
    //1,1,1,1,

    //Result:

    //1,4,0,0,
    //0,2,2,2,
    //0,1,0,1,

    //0,6,0,0,
    //2,5,-3,0,
    //1,4,1,1,


    //NCHW
    vector<float> expected({
        1, 4, 0, 0,
        0, 2, 2, 2,
        0, 1, 0, 1,

        0, 6, 0, 0,
        2, 5, -3, 0,
        1, 4, 1, 1,
    });

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
  }

}

static void testGlobalPoolingResidualBlock(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const GlobalPoolingResidualBlockDesc& desc, const vector<float>& input, const vector<float>& mask, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : input;
        vector<float> maskThisLoop = mask;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,desc.preBN.numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateGlobalPoolingResidualBlock(
          &desc,batchSize,nnXLen,nnYLen,useFP16,useNHWC,inputThisLoop,maskThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.preBN.numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.preBN.numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    string label("Global pooling residual block");

    int batchSize = 2;
    int trunkChannels = 1;
    int regularChannels = 1;
    int gpoolChannels = 2;
    int nnYLen = 3;
    int nnXLen = 4;

    //NCHW
    vector<float> input({
      1,2,0,0,
      0,3,4,0,
      0,0,5,0,

      0,0,0,0,
      0,5,-3,0,
      0,-1,1,1,
    });

    vector<float> mask({
      1,1,1,0,
      1,1,1,0,
      1,1,1,0,

      0,0,0,0,
      0,1,1,1,
      0,1,1,1,
    });

    GlobalPoolingResidualBlockDesc desc;

    //Identity map
    desc.preBN.name = "preBN";
    desc.preBN.numChannels = trunkChannels;
    desc.preBN.epsilon = 0.1;
    desc.preBN.hasScale = true;
    desc.preBN.hasBias = true;
    desc.preBN.mean = vector<float>({0});
    desc.preBN.variance = vector<float>({0.9});
    desc.preBN.scale = vector<float>({1});
    desc.preBN.bias = vector<float>({0});

    //ReLU gets applied, smooshing negatives
    //1,2,0,0,
    //0,3,4,0,
    //0,0,5,0,

    //0,0,0,0,
    //0,5,0,0,
    //0,0,1,1,

    //Double the value
    desc.regularConv.name = "regularConv";
    desc.regularConv.convYSize = 1;
    desc.regularConv.convXSize = 1;
    desc.regularConv.inChannels = trunkChannels;
    desc.regularConv.outChannels = regularChannels;
    desc.regularConv.dilationY = 1;
    desc.regularConv.dilationX = 1;
    desc.regularConv.weights = vector<float>({
        2
    });
    //2,4,0,0,
    //0,6,8,0,
    //0,0,10,0,

    //0,0,0,0,
    //0,10,0,0,
    //0,0,2,2,

    //For gpooling, split into two channels, shifting left and right
    desc.gpoolConv.name = "gpoolConv";
    desc.gpoolConv.convYSize = 3;
    desc.gpoolConv.convXSize = 3;
    desc.gpoolConv.inChannels = trunkChannels;
    desc.gpoolConv.outChannels = gpoolChannels;
    desc.gpoolConv.dilationY = 1;
    desc.gpoolConv.dilationX = 1;
    desc.gpoolConv.weights = vector<float>({
        0,0,0,
        0,0,1,
        0,0,0,

        0,0,0,
        1,0,0,
        0,0,0,
    });
    //2,0,0,0,
    //3,4,0,0,
    //0,5,0,0,

    //0,1,2,0,
    //0,0,3,0,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,1,1,0,

    //0,0,0,0,
    //0,0,5,0,
    //0,0,0,1,

    //Subtract 2 from all values in the 1th channel
    desc.gpoolBN.name = "gpoolBN";
    desc.gpoolBN.numChannels = gpoolChannels;
    desc.gpoolBN.epsilon = 0.1;
    desc.gpoolBN.hasScale = false;
    desc.gpoolBN.hasBias = false;
    desc.gpoolBN.mean = vector<float>({0,0});
    desc.gpoolBN.variance = vector<float>({0.9,0.9});
    desc.gpoolBN.scale = vector<float>({1,1});
    desc.gpoolBN.bias = vector<float>({0,-2});

    //And apply RELU

    //2,0,0,0,
    //3,4,0,0,
    //0,5,0,0,

    //0,0,0,0,
    //0,0,1,0,
    //0,0,0,0,

    //0,0,0,0,
    //0,0,0,0,
    //0,1,1,0,

    //0,0,0,0,
    //0,0,3,0,
    //0,0,0,0,

    //Pooling - mean, mean * (sqrt(masksum) - 14) * 0.1, max

    //14.0/9.0, 14.0/9.0*(-11)*0.1, 5
    //1.0/9.0, 1.0/9.0*(-11)*0.1, 1

    //2.0/6.0, 2.0/6.0*(sqrt(6)-14)*0.1, 1
    //3.0/6.0, 3.0/6.0*(sqrt(6)-14)*0.1, 3

    //Recombine values
    desc.gpoolToBiasMul.inChannels = 6;
    desc.gpoolToBiasMul.outChannels = regularChannels;
    desc.gpoolToBiasMul.weights = vector<float>({36,36, 18,18, 1,1});

    //56 + 28*(-11)*0.1 + 5 +
    //4 + 2*(-11)*0.1 + 1

    //12 + 6*(sqrt(6)-14)*0.1 + 1 +
    //18 + 9*(sqrt(6)-14)*0.1 + 3

    //Identity map
    desc.midBN.name = "midBN";
    desc.midBN.numChannels = regularChannels;
    desc.midBN.epsilon = 0.1;
    desc.midBN.hasScale = false;
    desc.midBN.hasBias = false;
    desc.midBN.mean = vector<float>({0});
    desc.midBN.variance = vector<float>({0.9});
    desc.midBN.scale = vector<float>({1});
    desc.midBN.bias = vector<float>({0});

    //Relu gets applied, should hit nothing in this case

    //Identity map
    desc.finalConv.name = "finalConv";
    desc.finalConv.convYSize = 1;
    desc.finalConv.convXSize = 1;
    desc.finalConv.inChannels = regularChannels;
    desc.finalConv.outChannels = trunkChannels;
    desc.finalConv.dilationY = 1;
    desc.finalConv.dilationX = 1;
    desc.finalConv.weights = vector<float>({
        1
    });

    vector<float> expected({
      3,6,0,0,
      0,9,12,0,
      0,0,15,0,

      0,0,0,0,
      0,15,-3,0,
      0,-1,3,3,
    });

    for(int i = 0; i<12; i++) {
      expected[i] +=
        56 + 28*(-11)*0.1 + 5 +
        4 + 2*(-11)*0.1 + 1;
      expected[i] *= mask[i];
    }
    for(int i = 12; i<24; i++) {
      expected[i] +=
        12 + 6*(sqrt(6)-14)*0.1 + 1 +
        18 + 9*(sqrt(6)-14)*0.1 + 3;
      expected[i] *= mask[i];
    }

    testConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,mask,expected);
  }

}

static void testSymmetries(int64_t& numTestsRun) {

  auto testConfigurations = [&](
    const string& label,
    int batchSize, int numChannels, int nnXLen, int nnYLen,
    const bool* symmetries,
    const vector<float>& input, const vector<float>& expected
  ) {
    for(int useNHWC = 0; useNHWC <= 1; useNHWC++) {
      for(int useFP16 = 0; useFP16 <= 1; useFP16++) {
        vector<float> inputThisLoop = useNHWC ? NCHWtoNHWC(input,batchSize,numChannels,nnYLen,nnXLen) : input;
        vector<float> expectedThisLoop = useNHWC ? NCHWtoNHWC(expected,batchSize,numChannels,nnYLen,nnXLen) : expected;

        vector<float> outputThisLoop;
        bool supported = NeuralNet::testEvaluateSymmetry(
          batchSize,numChannels,nnXLen,nnYLen,useFP16,useNHWC,symmetries,inputThisLoop,outputThisLoop
        );

        if(supported) {
          numTestsRun += 1;
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,numChannels,useFP16);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,numChannels,nnYLen,nnXLen,useFP16);
        }
      }
    }
  };

  {
    int nnYLen = 3;
    int nnXLen = 3;

    //NCHW
    vector<float> input({
        0,1,2,
        3,4,5,
        6,7,8,

        3,0,4,
        0,5,0,
        0,6,0,

        1,0,0,
        1,1,1,
        1,0,1,
    });

    {
      string label("Symmetry 0");
      bool symmetries[3] = {false,false,false};
      //NCHW
      vector<float> expected({
        0,1,2,
        3,4,5,
        6,7,8,

        3,0,4,
        0,5,0,
        0,6,0,

        1,0,0,
        1,1,1,
        1,0,1,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

    {
      string label("Symmetry 1");
      bool symmetries[3] = {true,false,false};
      //NCHW
      vector<float> expected({
        6,7,8,
        3,4,5,
        0,1,2,

        0,6,0,
        0,5,0,
        3,0,4,

        1,0,1,
        1,1,1,
        1,0,0,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

    {
      string label("Symmetry 2");
      bool symmetries[3] = {false,true,false};
      //NCHW
      vector<float> expected({
        2,1,0,
        5,4,3,
        8,7,6,

        4,0,3,
        0,5,0,
        0,6,0,

        0,0,1,
        1,1,1,
        1,0,1,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

    {
      string label("Symmetry 3");
      bool symmetries[3] = {true,true,false};
      //NCHW
      vector<float> expected({
        8,7,6,
        5,4,3,
        2,1,0,

        0,6,0,
        0,5,0,
        4,0,3,

        1,0,1,
        1,1,1,
        0,0,1,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

    {
      string label("Symmetry 4");
      bool symmetries[3] = {false,false,true};
      //NCHW
      vector<float> expected({
        0,3,6,
        1,4,7,
        2,5,8,

        3,0,0,
        0,5,6,
        4,0,0,

        1,1,1,
        0,1,0,
        0,1,1,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

    {
      string label("Symmetry 6");
      bool symmetries[3] = {false,true,true};
      //NCHW
      vector<float> expected({
        2,5,8,
        1,4,7,
        0,3,6,

        4,0,0,
        0,5,6,
        3,0,0,

        0,1,1,
        0,1,0,
        1,1,1,
      });
      testConfigurations(label,3,1,nnXLen,nnYLen,symmetries,input,expected);
      testConfigurations(label,1,3,nnXLen,nnYLen,symmetries,input,expected);
    }

  }

}



void Tests::runNNLayerTests() {
  NeuralNet::globalInitialize();
  int64_t numTestsRun = 0;
  testConvLayer(numTestsRun);
  testBatchNormLayer(numTestsRun);
  testResidualBlock(numTestsRun);
  testGlobalPoolingResidualBlock(numTestsRun);
  testSymmetries(numTestsRun);
  NeuralNet::globalCleanup();
  cout << "Tested " << numTestsRun << " configurations" << endl;
  cout << "Done" << endl;
}
