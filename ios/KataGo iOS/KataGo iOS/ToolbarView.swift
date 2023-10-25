//
//  ToolbarView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/10/1.
//

import SwiftUI

struct ToolbarView: View {
    @EnvironmentObject var player: PlayerObject
    @EnvironmentObject var config: Config
    
    var body: some View {
        HStack {
            Button(action: {
                KataGoHelper.sendCommand("undo")
                KataGoHelper.sendCommand("showboard")
                if config.isAnalyzing {
                    KataGoHelper.sendCommand(config.getKataAnalyzeCommand())
                }
            }) {
                Image(systemName: "arrow.uturn.backward")
            }
            .padding()

            Button(action: {
                let nextColor = (player.nextColorForPlayCommand == .black) ? "b" : "w"
                let pass = "play \(nextColor) pass"
                KataGoHelper.sendCommand(pass)
                KataGoHelper.sendCommand("showboard")
                if config.isAnalyzing {
                    KataGoHelper.sendCommand(config.getKataAnalyzeCommand())
                }
            }) {
                Image(systemName: "hand.raised")
            }
            .padding()

            Button(action: {
                if config.isAnalyzing {
                    KataGoHelper.sendCommand(config.getKataAnalyzeCommand())
                }
            }) {
                Image(systemName: "play")
            }
            .padding()

            Button(action: {
                if config.isAnalyzing {
                    KataGoHelper.sendCommand("stop")
                }
            }) {
                Image(systemName: "stop")
            }
            .padding()

            Button(action: {
                KataGoHelper.sendCommand("clear_board")
                KataGoHelper.sendCommand("showboard")
                if config.isAnalyzing {
                    KataGoHelper.sendCommand(config.getKataAnalyzeCommand())
                }
            }) {
                Image(systemName: "clear")
            }
            .padding()
        }
    }
}

struct ToolbarView_Previews: PreviewProvider {
    static let player = PlayerObject()
    static let config = Config()

    static var previews: some View {
        @State var isAnalyzing = true
        ToolbarView()
            .environmentObject(player)
            .environmentObject(config)
    }
}
