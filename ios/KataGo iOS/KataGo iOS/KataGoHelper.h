//
//  KataGoHelper.h
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

#ifndef KataGoHelper_h
#define KataGoHelper_h

#import <Foundation/Foundation.h>

@interface KataGoHelper : NSObject

+ (void)runGtp;

+ (NSString * _Nonnull)getMessageLine;

+ (void)sendCommand:(NSString * _Nonnull)command;

@end

#endif /* KataGoHelper_h */
