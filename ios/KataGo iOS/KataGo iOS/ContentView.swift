//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

/// Message with a text and an ID
struct Message: Identifiable, Equatable, Hashable {
    /// Identification of this message
    let id = UUID()

    /// Text of this message
    let text: String

    /// Initialize a message with a text
    /// - Parameter text: a text
    init(text: String) {
        self.text = text
    }
}

struct ContentView: View {
    @State private var messages: [Message] = []
    @State private var command = ""

    init() {
        // Start a thread to run KataGo GTP
        Thread {
            KataGoHelper.runGtp()
        }.start()
    }

    var body: some View {
        VStack {
            ScrollViewReader { scrollView in
                ScrollView(.vertical) {
                    // Vertically show each KataGo message
                    LazyVStack {
                        ForEach(messages) { message in
                            Text(message.text)
                                .font(.body.monospaced())
                                .id(message.id)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .onChange(of: messages) { value in
                        // Scroll to the last message
                        scrollView.scrollTo(value.last?.id)
                    }
                }
                .onAppear() {
                    // Get messages from KataGo and append to the list of messages
                    createMessageTask()
                }
            }

            HStack {
                TextField("Enter your GTP command", text: $command)
                    .disableAutocorrection(true)
                    .textInputAutocapitalization(.never)
                    .onSubmit {
                        messages.append(Message(text: command))
                        KataGoHelper.sendCommand(command)
                        command = ""
                    }
                Button(action: {
                    messages.append(Message(text: command))
                    KataGoHelper.sendCommand(command)
                    command = ""
                }) {
                    Image(systemName: "return")
                }
            }
            .padding()
        }
        .padding()
    }

    /// Create message task
    private func createMessageTask() {
        Task {
            while true {
                // Get a message line from KataGo
                let line = await KataGoHelper.messageLine()

                // Create a message with the line
                let message = Message(text: line)

                // Append the message to the list of messages
                messages.append(message)
            }
        }
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
