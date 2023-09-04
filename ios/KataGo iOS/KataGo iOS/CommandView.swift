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
    @EnvironmentObject var messagesObject: MessagesObject
    @State private var command = ""

    var body: some View {
        VStack {
            ScrollViewReader { scrollView in
                ScrollView(.vertical) {
                    // Vertically show each KataGo message
                    LazyVStack {
                        ForEach(messagesObject.messages) { message in
                            Text(message.text)
                                .font(.body.monospaced())
                                .id(message.id)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                    .onChange(of: messagesObject.messages) { value in
                        // Scroll to the last message
                        scrollView.scrollTo(value.last?.id)
                    }
                }
            }

            HStack {
                TextField("Enter your GTP command (list_commands)", text: $command)
                    .disableAutocorrection(true)
                    .textInputAutocapitalization(.never)
                    .onSubmit {
                        messagesObject.messages.append(Message(text: command))
                        KataGoHelper.sendCommand(command)
                        command = ""
                    }
                Button(action: {
                    messagesObject.messages.append(Message(text: command))
                    KataGoHelper.sendCommand(command)
                    command = ""
                }) {
                    Image(systemName: "return")
                }
            }
            .padding()

            HStack {
                CommandButton(title: "genmove b") {
                    messagesObject.messages.append(Message(text: "genmove b"))
                    KataGoHelper.sendCommand("genmove b")
                }

                CommandButton(title: "genmove w") {
                    messagesObject.messages.append(Message(text: "genmove w"))
                    KataGoHelper.sendCommand("genmove w")
                }

                CommandButton(title: "showboard") {
                    messagesObject.messages.append(Message(text: "showboard"))
                    KataGoHelper.sendCommand("showboard")
                }

                CommandButton(title: "clear_board") {
                    messagesObject.messages.append(Message(text: "clear_board"))
                    KataGoHelper.sendCommand("clear_board")
                }
            }
        }
        .padding()
    }
}

struct CommandView_Previews: PreviewProvider {
    static let messageObject = MessagesObject()

    static var previews: some View {
        CommandView()
            .environmentObject(messageObject)
    }
}
