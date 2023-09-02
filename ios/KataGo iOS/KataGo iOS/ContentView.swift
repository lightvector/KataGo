//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

struct ContentView: View {
    @State private var selection: Tab = .command

    enum Tab {
        case command
        case goban
    }

    var body: some View {
        TabView(selection: $selection) {
            CommandView()
                .tabItem {
                    Label("Command", systemImage: "text.alignleft")
                }
                .tag(Tab.command)


            GobanView()
                .tabItem {
                    Label("Goban", systemImage: "circle")
                }
                .tag(Tab.goban)
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
