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
        let totalLength = min(totalWidth, totalHeight)
        let boardSpace: CGFloat = totalLength * 0.05
        let squareWidth = (totalWidth - boardSpace) / (width + 1)
        let squareHeight = (totalHeight - boardSpace) / (height + 1)
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

    var body: some View {
        GeometryReader { geometry in
            let dimensions = Dimensions(geometry: geometry, width: board.width, height: board.height)
            ZStack {
                BoardLineView(dimensions: dimensions, boardWidth: board.width, boardHeight: board.height)
                StoneView(dimensions: dimensions)
                AnalysisView(dimensions: dimensions)
            }
        }
        .gesture(TapGesture().onEnded() { _ in
            if nextPlayer.color == .black {
                KataGoHelper.sendCommand("genmove b")
                nextPlayer.color = .white
            } else {
                KataGoHelper.sendCommand("genmove w")
                nextPlayer.color = .black
            }

            KataGoHelper.sendCommand("showboard")
            KataGoHelper.sendCommand("kata-analyze interval 10")
        })
        .onAppear() {
            KataGoHelper.sendCommand("showboard")
            KataGoHelper.sendCommand("kata-analyze interval 10")
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
