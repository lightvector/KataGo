//
//  KataGo_iOSApp.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

@main
struct KataGo_iOSApp: App {
    init() {
        DispatchQueue.global(qos: .background).async {
            KataGoHelper.runGtp()
        }
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
