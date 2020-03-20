#include "main.h"
#include "core/os.h"

// REFACT this could be the central location to include tclap and define this convention
// as this is extra define is currently replicated all over the source
#define TCLAP_NAMESTARTSTRING "-" //Use single dashes for all flags
#include <tclap/CmdLine.h>

#if defined(OS_IS_UNIX_OR_APPLE)
    #include <wordexp.h>
#elif defined(OS_IS_WINDOWS)
    // TODO whatever windows needs to expand the path to the home directory
#else
    #error Unknown operating system!
#endif

#include <boost/filesystem.hpp>
namespace bfs = boost::filesystem;


using namespace std;

class KataGoCommandLine : public TCLAP::CmdLine
{
public:
    TCLAP::ValueArg<string> modelFileArg;
    TCLAP::ValueArg<string> configFileArg;
    
    KataGoCommandLine(const std::string& message)
        :
            TCLAP::CmdLine(message, ' ', Version::getKataGoVersionForHelp(),true),
            modelFileArg("","model","Neural net model file. Defaults to: " + defaultModelPath(), 
                !hasDefaultModelPath(), defaultModelPath(),"FILE"),
            configFileArg("","config","Config file to use (see configs/*_example.cfg). Defaults to: " + defaultConfigPath(), 
                !hasDefaultConfigPath(),defaultConfigPath(),"FILE")
    {
    }
    
#pragma mark modelFileArg
    
    void addModelFileArg() {
        this->add(this->modelFileArg);
    }
    
    static std::string defaultModelPath() {
        return defaultPathIfItExists("default_model.bin.gz");
    }
    
    static bool hasDefaultModelPath() {
        return ! defaultModelPath().empty();
    }
    
#pragma mark configFileArg
    
    void addConfigFileArg() {
        this->add(this->configFileArg);
    }
    
    static std::string defaultConfigPath() {
        return defaultPathIfItExists("default_config.cfg");
    }
    
    static bool hasDefaultConfigPath() {
        return ! defaultModelPath().empty();
    }
    
    
#pragma mark Support code
    
    static bfs::path getHomeDirectory() {
        bfs::path homeDirectory;
#if defined(OS_IS_WINDOWS)
        #error FIXME needs implementing
        // TODO I have no clue how windows handles this
        // possibly using ExpandEnvironmentString, see https://stackoverflow.com/questions/1902681/expand-file-names-that-have-environment-variables-in-their-path
#elif defined(OS_IS_UNIX_OR_APPLE)
        wordexp_t expandedPath;
        wordexp("~", &expandedPath, 0);
        homeDirectory = expandedPath.we_wordv[0];
        wordfree(&expandedPath);
#else
        #error Unknown operating system!
#endif
        return homeDirectory;
    }
    
    static std::string defaultPathIfItExists(bfs::path standardFileName) {
        bfs::path homeDirectory = getHomeDirectory();
        bfs::path standardModelPath = homeDirectory / ".katago" / standardFileName;
        if ( bfs::exists(standardModelPath)) {
            return standardModelPath.native();
        }
        
        // no default file found
        return std::string();
    }
};
