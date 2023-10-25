//
//  ButtonView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/5.
//

import SwiftUI

struct ButtonView: View {
    @EnvironmentObject var messagesObject: MessagesObject
    @EnvironmentObject var config: Config
    let commands: [String]
    
    var body: some View {
        HStack {
            ForEach(commands, id:\.self) { command in
                CommandButton(title: command) {
                    messagesObject.messages.append(Message(text: command, maxLength: config.maxMessageCharacters))
                    KataGoHelper.sendCommand(command)
                }
                .scaledToFit()
            }
        }
    }
}

struct ButtonView_Previews: PreviewProvider {
    static let commands = ["kata-set-rules chinese", "komi 7"]
    static var messagesObject = MessagesObject()

    static var previews: some View {
        ButtonView(commands: commands)
            .environmentObject(messagesObject)
    }
}
