//
//  GobanView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/2.
//

import SwiftUI

struct Dimensions {
    let squareLength: CGFloat
    let boardWidth: CGFloat
    let boardHeight: CGFloat
    let marginWidth: CGFloat
    let marginHeight: CGFloat

    init(geometry: GeometryProxy, width: CGFloat, height: CGFloat) {
        let totalWidth = geometry.size.width
        let totalHeight = geometry.size.height
        let squareWidth = totalWidth / (width + 1)
        let squareHeight = totalHeight / (height + 1)
        squareLength = min(squareWidth, squareHeight)
        boardWidth = width * squareLength
        boardHeight = height * squareLength
        marginWidth = (totalWidth - boardWidth + squareLength) / 2
        marginHeight = (totalHeight - boardHeight + squareLength) / 2
    }
}

struct GobanView: View {
    @EnvironmentObject var stones: Stones
    @EnvironmentObject var board: Board
    @EnvironmentObject var player: PlayerObject
    @EnvironmentObject var analysis: Analysis
    @State var isAnalyzing = true
    let texture = WoodImage.createTexture()
    let kataAnalyze = "kata-analyze interval 20 maxmoves 32 ownership true ownershipStdev true"

    var body: some View {
        VStack {
            HStack {
                Toggle(isOn: $isAnalyzing) {
                    Text("Analysis")
                }
                .onChange(of: isAnalyzing) { flag in
                    if flag {
                        KataGoHelper.sendCommand(kataAnalyze)
                    } else {
                        KataGoHelper.sendCommand("stop")
                    }
                }
            }
            .padding()

            GeometryReader { geometry in
                let dimensions = Dimensions(geometry: geometry, width: board.width, height: board.height)
                ZStack {
                    BoardLineView(dimensions: dimensions, boardWidth: board.width, boardHeight: board.height)
                    StoneView(dimensions: dimensions)
                    if isAnalyzing {
                        AnalysisView(dimensions: dimensions)
                    }
                }
                .onTapGesture(coordinateSpace: .local) { location in
                    if let move = locationToMove(location: location, dimensions: dimensions) {
                        if player.nextPlay == .black {
                            KataGoHelper.sendCommand("play b \(move)")
                            player.nextPlay = .white
                        } else {
                            KataGoHelper.sendCommand("play w \(move)")
                            player.nextPlay = .black
                        }
                    }

                    KataGoHelper.sendCommand("showboard")
                    if isAnalyzing {
                        KataGoHelper.sendCommand(kataAnalyze)
                    }
                }
            }
            .onAppear() {
                KataGoHelper.sendCommand("showboard")
                if isAnalyzing {
                    KataGoHelper.sendCommand(kataAnalyze)
                }
            }

            HStack {
                Button(action: {
                    KataGoHelper.sendCommand("undo")
                    KataGoHelper.sendCommand("showboard")
                    if isAnalyzing {
                        KataGoHelper.sendCommand(kataAnalyze)
                    }
                }) {
                    Image(systemName: "arrow.uturn.backward")
                }
                Button(action: {
                    let nextColor = (player.nextPlay == .black) ? "b" : "w"
                    let pass = "play \(nextColor) pass"
                    KataGoHelper.sendCommand(pass)
                    KataGoHelper.sendCommand("showboard")
                    if isAnalyzing {
                        KataGoHelper.sendCommand(kataAnalyze)
                    }
                }) {
                    Image(systemName: "hand.raised")
                }
                Button(action: {
                    if isAnalyzing {
                        KataGoHelper.sendCommand(kataAnalyze)
                    }
                }) {
                    Image(systemName: "play")
                }
                Button(action: {
                    if isAnalyzing {
                        KataGoHelper.sendCommand("stop")
                    }
                }) {
                    Image(systemName: "stop")
                }
                Button(action: {
                    KataGoHelper.sendCommand("clear_board")
                    KataGoHelper.sendCommand("showboard")
                    if isAnalyzing {
                        KataGoHelper.sendCommand(kataAnalyze)
                    }
                }) {
                    Image(systemName: "clear")
                }
            }
            .padding()
        }
    }

    func locationToMove(location: CGPoint, dimensions: Dimensions) -> String? {
        let x = Int(round((location.x - dimensions.marginWidth) / dimensions.squareLength))
        let y = Int(round((location.y - dimensions.marginHeight) / dimensions.squareLength)) + 1

        // Mapping 0-18 to letters A-T (without I)
        let letterMap: [Int: String] = [
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
            5: "F", 6: "G", 7: "H", 8: "J", 9: "K",
            10: "L", 11: "M", 12: "N", 13: "O", 14: "P",
            15: "Q", 16: "R", 17: "S", 18: "T"
        ]

        if let letter = letterMap[x] {
            let move = "\(letter)\(y)"
            return move
        } else {
            return nil
        }
    }
}

struct GobanView_Previews: PreviewProvider {
    static let stones = Stones()
    static let board = Board()
    static let analysis = Analysis()
    static let player = PlayerObject()

    static var previews: some View {
        GobanView()
            .environmentObject(stones)
            .environmentObject(board)
            .environmentObject(analysis)
            .environmentObject(player)
            .onAppear() {
                GobanView_Previews.board.width = 3
                GobanView_Previews.board.height = 3
                GobanView_Previews.stones.blackPoints = [BoardPoint(x: 1, y: 1), BoardPoint(x: 0, y: 1)]
                GobanView_Previews.stones.whitePoints = [BoardPoint(x: 0, y: 0), BoardPoint(x: 1, y: 0)]
                GobanView_Previews.analysis.data = [["move": "C1", "winrate": "0.54321012345", "visits": "1234567890", "scoreLead": "8.987654321"]]
            }
    }
}
