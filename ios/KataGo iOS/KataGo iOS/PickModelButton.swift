//
//  PickModelButton.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/10/10.
//

import SwiftUI

struct PickModelButton: View {
    static let defaultFileURL = KataGoHelper.getAppMLModelURL()

    @Environment(\.editMode) private var editMode
    @State private var selectedFileURL = defaultFileURL
    @State private var showFileImporter = false

    var body: some View {
        HStack {
            Text("Update model:")
            Spacer()
            Text(selectedFileURL?.absoluteString ?? "Cannot create Application ML Model URL!")
                .onTapGesture {
                    if editMode?.wrappedValue.isEditing == true {
                        showFileImporter = true
                    }
                }
                .fileImporter(
                    isPresented: $showFileImporter,
                    allowedContentTypes: [.directory],
                    allowsMultipleSelection: false
                ) { result in
                    if let defaultURL = PickModelButton.defaultFileURL {
                        switch result {
                        case .success(let urls):
                            if let url = urls.first {
                                do {
                                    try FileManager.default.removeItem(at: defaultURL)
                                    try FileManager.default.copyItem(at: url, to: defaultURL)

                                    selectedFileURL = url
                                } catch {
                                    print(error)
                                }
                            }
                        case .failure(let error):
                            // handle error
                            print(error)
                        }
                    }
                }
                .background((editMode?.wrappedValue.isEditing ?? false) ? Color(white: 0.9) : .clear)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

#Preview {
    PickModelButton()
}
