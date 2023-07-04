//
//  KataGoHelper.m
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

#import "KataGoHelper.h"
#import "../../cpp/main.h"
#import <sstream>

using namespace std;

// Thread-safe stream buffer
class ThreadSafeStreamBuf : public std::streambuf {
    std::string buffer;
    std::mutex m;
    std::condition_variable cv;
    std::atomic<bool> done {false};

public:
    int overflow(int c) override {
        std::lock_guard<std::mutex> lock(m);
        buffer += static_cast<char>(c);
        if (c == '\n') {
            cv.notify_all();
        }
        return c;
    }

    int underflow() override {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]{ return !buffer.empty() || done; });
        if (buffer.empty()) {
            return std::char_traits<char>::eof();
        }
        return buffer.front();
    }

    int uflow() override {
        std::unique_lock<std::mutex> lock(m);
        cv.wait(lock, [&]{ return !buffer.empty() || done; });
        if (buffer.empty()) {
            return std::char_traits<char>::eof();
        }
        int c = buffer.front();
        buffer.erase(buffer.begin());
        return c;
    }

    void setDone() {
        done = true;
        cv.notify_all();
    }
};

// Thread-safe stream buffer from KataGo
ThreadSafeStreamBuf tsbFromKataGo;

// Input stream from KataGo
istream inFromKataGo(&tsbFromKataGo);

@implementation KataGoHelper

/// Run KataGo main command GTP with default model and config
+ (void)runGtp {
    NSBundle* mainBundle = [NSBundle mainBundle];

    // Get the default model path
    NSString* modelPath = [mainBundle pathForResource:@"default_model"
                                               ofType:@"bin.gz"];

    // Get the default config path
    NSString* configPath = [mainBundle pathForResource:@"default_gtp"
                                                ofType:@"cfg"];

    // Replace the global cout object with the custom one
    cout.rdbuf(&tsbFromKataGo);

    vector<string> subArgs;
#if false
    // Call the main command gtp
    subArgs.push_back(string("gtp"));
    subArgs.push_back(string("-model"));
    subArgs.push_back(string([modelPath UTF8String]));
    subArgs.push_back(string("-config"));
    subArgs.push_back(string([configPath UTF8String]));
    MainCmds::gtp(subArgs);
#else
    // Call the main command benchmark
    subArgs.push_back(string("benchmark"));
    subArgs.push_back(string("-model"));
    subArgs.push_back(string([modelPath UTF8String]));
    subArgs.push_back(string("-config"));
    subArgs.push_back(string([configPath UTF8String]));
    subArgs.push_back(string("-t"));
    subArgs.push_back(string("2,4,8"));
    MainCmds::benchmark(subArgs);
#endif
}

+ (void)getOneMessageLineWithCompletion:(void (^ _Nullable)(NSString * _Nonnull messageLine))completion {
    // Get a line from the input stream from KataGo
    string cppLine;
    getline(inFromKataGo, cppLine);

    // Convert the C++ std:string into an NSString
    NSString* messageLine = [NSString stringWithUTF8String:cppLine.c_str()];

    completion(messageLine);
}

@end
