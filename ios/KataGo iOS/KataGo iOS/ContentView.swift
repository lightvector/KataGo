//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

struct BoardPoint: Hashable {
    let x: Int
    let y: Int
}

class Stones: ObservableObject {
    @Published var blackPoints: [BoardPoint] = []
    @Published var whitePoints: [BoardPoint] = []
}

class MessagesObject: ObservableObject {
    @Published var messages: [Message] = []
}

struct ContentView: View {
    @StateObject var stones: Stones = Stones()
    @StateObject private var messagesObject: MessagesObject = MessagesObject()
    @State private var selection: Tab = .command

    enum Tab {
        case command
        case goban
    }

    init() {
        // Start a thread to run KataGo GTP
        Thread {
            KataGoHelper.runGtp()
        }.start()
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
        .environmentObject(stones)
        .environmentObject(messagesObject)
        .onAppear() {
            // Get messages from KataGo and append to the list of messages
            createMessageTask()
        }
    }

    /// Create message task
    private func createMessageTask() {
        Task {
            messagesObject.messages.append(Message(text: "Initializing..."))
            KataGoHelper.sendCommand("showboard")
            while true {
                let line = await Task.detached {
                    // Get a message line from KataGo
                    return await KataGoHelper.messageLine()
                }.value

                // Create a message with the line
                let message = Message(text: line)

                // Append the message to the list of messages
                messagesObject.messages.append(message)

                // TODO: Update `stones` here
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
