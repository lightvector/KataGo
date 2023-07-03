//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

/// Message ID actor. Actor allows only one task to access the mutable state at a time.
actor MessageId {
    var value: Int;

    /// Initialize a message ID with a value
    /// - Parameter value: a value
    init(_ value: Int) {
        self.value = value
    }

    /// Increment the message ID
    /// - Returns: the incremented value
    func increment() -> Int {
        value = value + 1
        return value
    }
}

/// Message with a text and an ID
struct Message: Identifiable, Equatable, Hashable {
    private static var lastId = MessageId(-1)

    /// Get the next ID, which is increased by 1
    /// - Returns: the next ID
    static func getNextId() async -> Int {
        return await lastId.increment()
    }

    /// Get the last ID
    /// - Returns: the last ID
    static func getLastId() async -> Int {
        return await lastId.value
    }

    /// Identification of this message
    let id: Int

    /// Text of this message
    let text: String

    /// Initialize a message with a text
    /// - Parameter text: a text
    init(text: String) async {
        self.id = await Message.getNextId()
        self.text = text
    }
}

/// KataGo controller
class KataGoController: ObservableObject {
    /// A list of messages
    @Published var messages: [Message] = []

    /// Get the ID of the last message
    /// - Returns: the ID of the last message
    func getLastID() async -> Int {
        return await Message.getLastId()
    }

    /// Process a message from KataGo
    func processMessage() {
        // Get a message line from KataGo
        let line = KataGoHelper.getMessageLine()

        Task.detached {
            // Create a message with the line
            let message = await Message(text: line)

            // Append the message to the list of messages
            DispatchQueue.main.async {
                self.messages.append(message)
            }
        }
    }

    /// Process messages from KataGo
    func processMessages() {
        while (true) {
            processMessage()
        }
    }

    /// Start a thread to process messages from KataGo
    func startMessageThread() {
        Thread {
            self.processMessages()
        }.start()
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
                        if let id = value.last?.id {
                            scrollView.scrollTo(id)
                        }
                    }
                }
            }
            .onAppear() {
                kataGo.startMessageThread()
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
