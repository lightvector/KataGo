//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

/// Message with a text and an ID
struct Message: Identifiable, Equatable, Hashable {
    static var id = -1

    static func getID() -> Int {
        id += 1
        return id
    }

    let id = getID()
    let text: String
}

/// KataGo controller
class KataGoController: ObservableObject {
    @Published var messages: [Message] = []

    /// Get the ID of the last message
    /// - Returns: the ID of the last message
    func getLastID() -> Int {
        return messages[messages.endIndex - 1].id
    }

    func waitMessageAndUpdate() {
        // Wait until a message line is available
        let line = KataGoHelper.getMessageLine()
        let message = Message(text: line)

        // Update the messages
        DispatchQueue.main.async {
            self.messages.append(message)
        }
    }
}

struct ContentView: View {
    @ObservedObject private var kataGo = KataGoController()

    var body: some View {
        VStack {
            ScrollViewReader { scrollView in
                ScrollView(.vertical) {
                    // Vertically show each KataGo message
                    LazyVStack {
                        ForEach(kataGo.messages) { message in
                            Text(message.text)
                                .padding()
                                .id(message.id)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .onChange(of: kataGo.messages) { value in
                        // Scroll to the last message
                        if value.count > 0 {
                            scrollView.scrollTo(kataGo.getLastID())
                        }
                    }
                }
            }
            .onAppear() {
                // Start a thread to run an infinite loop that waits and updates KataGo messages
                Thread {
                    while (true) {
                        kataGo.waitMessageAndUpdate()
                    }
                }.start()
            }
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
