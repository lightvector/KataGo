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

+ (nonnull NSString*)getMessageLine;

@end

#endif /* KataGoHelper_h */
