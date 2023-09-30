//
//  ConfigView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/19.
//

import SwiftUI

struct ConfigView: View {
    @EnvironmentObject var config: Config
    @State var maxMessageCharacters: String = "200"
    @State var maxAnalysisMoves: String = "8"

    var body: some View {
        VStack {
            HStack {
                Text("Max message characters:")
                TextField("200", text: $maxMessageCharacters)
            }

            HStack {
                Text("Max analysis moves:")
                TextField("8", text: $maxAnalysisMoves)
            }
        }
        .padding()
        .onDisappear() {
            config.maxMessageCharacters = Int(maxMessageCharacters) ?? Config.defaultMaxMessageCharacters
            config.maxAnalysisMoves = Int(maxAnalysisMoves) ?? Config.defaultMaxAnalysisMoves
        }
    }
}

struct ConfigView_Previews: PreviewProvider {
    static let config = Config()
    static var previews: some View {
        ConfigView()
            .environmentObject(config)
    }
}
