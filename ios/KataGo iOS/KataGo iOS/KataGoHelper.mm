//
//  KataGoHelper.m
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

#import "KataGoHelper.h"
#import "../../cpp/main.h"

using namespace std;

@implementation KataGoHelper

+ (void)runGtp {
    NSBundle* mainBundle = [NSBundle mainBundle];

    NSString* modelPath = [mainBundle pathForResource:@"default_model"
                                               ofType:@"bin.gz"];

    NSString* configPath = [mainBundle pathForResource:@"default_gtp"
                                                ofType:@"cfg"];

    // Call the main command gtp
    vector<string> subArgs;
    subArgs.push_back(string("gtp"));
    subArgs.push_back(string("-model"));
    subArgs.push_back(string([modelPath UTF8String]));
    subArgs.push_back(string("-config"));
    subArgs.push_back(string([configPath UTF8String]));
    MainCmds::gtp(subArgs);
}

@end
