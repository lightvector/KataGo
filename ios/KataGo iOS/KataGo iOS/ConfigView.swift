//
//  ConfigView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/19.
//

import SwiftUI

struct EditButtonBar: View {
    var body: some View {
        HStack {
            Spacer()
            EditButton()
        }
    }
}

struct ConfigItem: View {
    @Environment(\.editMode) private var editMode
    let title: String
    @Binding var content: String

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            if editMode?.wrappedValue.isEditing == true {
                TextField("", text: $content)
                    .multilineTextAlignment(.trailing)
                    .background(Color(white: 0.9))
            } else {
                Text(content)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ConfigItems: View {
    @EnvironmentObject var config: Config
    @State var isAnalyzing = Config.defaultIsAnalyzing
    @State var maxMessageCharacters: String = "\(Config.defaultMaxMessageCharacters)"
    @State var maxAnalysisMoves: String = "\(Config.defaultMaxAnalysisMoves)"
    @State var analysisInterval: String = "\(Config.defaultAnalysisInterval)"
    @State var maxMessageLines: String = "\(Config.defaultMaxMessageLines)"

    var body: some View {
        VStack {
            HStack {
                Toggle(isOn: $isAnalyzing) {
                    Text("Analysis")
                }
                .onChange(of: isAnalyzing) { newFlag in
                    config.isAnalyzing = newFlag
                }
            }
            .padding(.bottom)

            ConfigItem(title: "Max message characters:", content: $maxMessageCharacters)
                .onChange(of: maxMessageCharacters) { newText in
                    config.maxMessageCharacters = Int(newText) ??
                    Config.defaultMaxMessageCharacters
                }
                .padding(.bottom)

            ConfigItem(title: "Max analysis moves:", content: $maxAnalysisMoves)
                .onChange(of: maxAnalysisMoves) { newText in
                    config.maxAnalysisMoves = Int(newText) ??
                    Config.defaultMaxAnalysisMoves
                }
                .padding(.bottom)

            ConfigItem(title: "Analysis interval (centiseconds):", content: $analysisInterval)
                .onChange(of: analysisInterval) { newText in
                    config.analysisInterval = Int(newText) ??
                    Config.defaultAnalysisInterval
                }
                .padding(.bottom)

            ConfigItem(title: "Max message lines:", content: $maxMessageLines)
                .onChange(of: maxMessageLines) { newText in
                    config.maxMessageLines = Int(newText) ??
                    Config.defaultMaxMessageLines
                }
        }
    }
}

struct ConfigView: View {
    var body: some View {
        VStack {
            EditButtonBar()
                .padding()
            ConfigItems()
                .padding()
        }
        .frame(maxHeight: .infinity, alignment: .topLeading)
        .onAppear() {
            KataGoHelper.sendCommand("stop")
        }
    }
}

struct ConfigView_Previews: PreviewProvider {
    static let isEditing = EditMode.inactive
    static let config = Config()
    static var previews: some View {
        ConfigView()
            .environmentObject(config)
    }
}
