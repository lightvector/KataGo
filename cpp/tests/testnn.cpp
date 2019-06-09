#include "../tests/tests.h"
#include "../neuralnet/nninterface.h"

#include <cmath>

using namespace std;

static bool approxEqual(float x, float y) {
  float tolerance = 0.001f * std::max(std::abs(x),std::max(std::abs(y),1.0f));
  return std::abs(x - y) < tolerance;
}

static void checkApproxEqual(
  const string& label,
  const vector<float>& vec, const vector<float>& expected, int nSize, int cSize, int ySize, int xSize,
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
          if(!approxEqual(vec[i],expected[i]) && !mismatch) {
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
#define CHECK_APPROX_EQUAL(label,vec,expected,n,c,h,w) (checkApproxEqual((label),(vec),(expected),(n),(c),(h),(w),__FILE__,#vec,__LINE__))


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


static void testConvLayer() {

  auto testConvConfigurations = [](
    const string& label,
    int batchSize, int nnXLen, int nnYLen,
    const ConvLayerDesc& desc, const vector<float> input, const vector<float> expected
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
          string subLabel = label + Global::strprintf(" useNHWC %d useFP16 %d", useNHWC, useFP16);
          if(useNHWC)
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,nnYLen,nnXLen,desc.outChannels);
          else
            CHECK_APPROX_EQUAL(subLabel,outputThisLoop,expectedThisLoop,batchSize,desc.outChannels,nnYLen,nnXLen);
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

      testConvConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
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

      testConvConfigurations(label,batchSize,nnXLen,nnYLen,desc,input,expected);
    }

  }


}

void Tests::runNNLayerTests() {
  testConvLayer();


}
