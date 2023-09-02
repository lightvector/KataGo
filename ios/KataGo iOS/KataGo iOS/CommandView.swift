//
//  CommandView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/2.
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

struct CommandButton: View {
    var title: String
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .foregroundColor(.white)
                .padding()
                .background(Color.blue)
                .clipShape(RoundedRectangle(cornerRadius: 50))
                .font(.body.monospaced())
        }
    }
}

struct CommandView: View {
    @State private var messages: [Message] = []
    @State private var command = ""
    @State private var running = false

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
            }
            .onAppear() {
                // Get messages from KataGo and append to the list of messages
                createMessageTask()
            }

            HStack {
                TextField("Enter your GTP command (list_commands)", text: $command)
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

            HStack {
                CommandButton(title: "genmove b") {
                    messages.append(Message(text: "genmove b"))
                    KataGoHelper.sendCommand("genmove b")
                }

                CommandButton(title: "genmove w") {
                    messages.append(Message(text: "genmove w"))
                    KataGoHelper.sendCommand("genmove w")
                }

                CommandButton(title: "showboard") {
                    messages.append(Message(text: "showboard"))
                    KataGoHelper.sendCommand("showboard")
                }

                CommandButton(title: "clear_board") {
                    messages.append(Message(text: "clear_board"))
                    KataGoHelper.sendCommand("clear_board")
                }
            }
        }
        .padding()
    }

    /// Create message task
    private func createMessageTask() {
        if !running {
            Task {
                running = true
                messages.append(Message(text: "Initializing..."))
                KataGoHelper.sendCommand("showboard")
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
}

struct CommandView_Previews: PreviewProvider {
    static var previews: some View {
        CommandView()
    }
}
