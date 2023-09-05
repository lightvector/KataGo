//
//  ButtonView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/5.
//

import SwiftUI

struct ButtonView: View {
    @EnvironmentObject var messagesObject: MessagesObject
    
    var body: some View {
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
}

struct ButtonView_Previews: PreviewProvider {
    static var previews: some View {
        ButtonView()
    }
}
