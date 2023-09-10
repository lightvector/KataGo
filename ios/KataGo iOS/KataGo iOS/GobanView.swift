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
    @EnvironmentObject var nextPlayer: PlayerObject
    @EnvironmentObject var analysis: Analysis
    let texture = WoodImage.createTexture()
    let kataAnalyze = "kata-analyze interval 10 ownership true ownershipStdev true"

    var body: some View {
        VStack {
            GeometryReader { geometry in
                let dimensions = Dimensions(geometry: geometry, width: board.width, height: board.height)
                ZStack {
                    BoardLineView(dimensions: dimensions, boardWidth: board.width, boardHeight: board.height)
                    StoneView(dimensions: dimensions)
                    AnalysisView(dimensions: dimensions)
                }
                .onTapGesture(coordinateSpace: .local) { location in
                    if let move = locationToMove(location: location, dimensions: dimensions) {
                        if nextPlayer.color == .black {
                            KataGoHelper.sendCommand("play b \(move)")
                            nextPlayer.color = .white
                        } else {
                            KataGoHelper.sendCommand("play w \(move)")
                            nextPlayer.color = .black
                        }
                    }

                    KataGoHelper.sendCommand("showboard")
                    KataGoHelper.sendCommand(kataAnalyze)
                }
            }
            .onAppear() {
                KataGoHelper.sendCommand("showboard")
                KataGoHelper.sendCommand(kataAnalyze)
            }

            ButtonView(commands: ["undo", "showboard", "stop", kataAnalyze])
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

    static var previews: some View {
        GobanView()
            .environmentObject(stones)
            .environmentObject(board)
            .environmentObject(analysis)
            .onAppear() {
                GobanView_Previews.stones.blackPoints = [BoardPoint(x: 15, y: 3), BoardPoint(x: 13, y: 2), BoardPoint(x: 9, y: 3), BoardPoint(x: 3, y: 3)]
                GobanView_Previews.stones.whitePoints = [BoardPoint(x: 3, y: 15)]
                GobanView_Previews.analysis.data = [["move": "Q16", "winrate": "0.54321012345"]]
            }
    }
}
