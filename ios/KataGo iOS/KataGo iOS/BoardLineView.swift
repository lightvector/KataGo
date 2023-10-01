//
//  BoardLineView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/9.
//

import SwiftUI

struct BoardLineView: View {
    let dimensions: Dimensions
    let texture = WoodImage.createTexture()
    let boardWidth: CGFloat
    let boardHeight: CGFloat

    var body: some View {
        ZStack {
            drawBoardBackground(texture: texture, dimensions: dimensions)
            drawLines(dimensions: dimensions)
            drawStarPoints(dimensions: dimensions)
        }
    }

    private func drawBoardBackground(texture: UIImage, dimensions: Dimensions) -> some View {
        Group {
            Image(uiImage: texture)
                .resizable()
                .frame(width: (dimensions.boardWidth + dimensions.squareLength / 2), height: dimensions.boardHeight + (dimensions.squareLength / 2))
        }
    }

    private func drawLines(dimensions: Dimensions) -> some View {
        Group {
            ForEach(0..<Int(boardHeight), id: \.self) { i in
                horizontalLine(i: i, dimensions: dimensions)
            }
            ForEach(0..<Int(boardWidth), id: \.self) { i in
                verticalLine(i: i, dimensions: dimensions)
            }
        }
    }

    private func horizontalLine(i: Int, dimensions: Dimensions) -> some View {
        Path { path in
            path.move(to: CGPoint(x: dimensions.marginWidth, y: dimensions.marginHeight + CGFloat(i) * dimensions.squareLength))
            path.addLine(to: CGPoint(x: dimensions.marginWidth + dimensions.boardWidth - dimensions.squareLength, y: dimensions.marginHeight + CGFloat(i) * dimensions.squareLength))
        }
        .stroke(Color.black)
    }

    private func verticalLine(i: Int, dimensions: Dimensions) -> some View {
        Path { path in
            path.move(to: CGPoint(x: dimensions.marginWidth + CGFloat(i) * dimensions.squareLength, y: dimensions.marginHeight))
            path.addLine(to: CGPoint(x: dimensions.marginWidth + CGFloat(i) * dimensions.squareLength, y: dimensions.marginHeight + dimensions.boardHeight - dimensions.squareLength))
        }
        .stroke(Color.black)
    }

    private func drawStarPoint(x: Int, y: Int, dimensions: Dimensions) -> some View {
        // Big black dot
        Circle()
            .frame(width: dimensions.squareLength / 4, height: dimensions.squareLength / 4)
            .foregroundColor(Color.black)
            .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                      y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
    }

    private func drawStarPointsForSize(points: [BoardPoint], dimensions: Dimensions) -> some View {
        ForEach(points, id: \.self) { point in
            drawStarPoint(x: point.x, y: point.y, dimensions: dimensions)
        }
    }

    private func drawStarPoints(dimensions: Dimensions) -> some View {
        Group {
            if boardWidth == 19 && boardHeight == 19 {
                // Draw star points for 19x19 board
                drawStarPointsForSize(points: [BoardPoint(x: 3, y: 3), BoardPoint(x: 3, y: 9), BoardPoint(x: 3, y: 15), BoardPoint(x: 9, y: 3), BoardPoint(x: 9, y: 9), BoardPoint(x: 9, y: 15), BoardPoint(x: 15, y: 3), BoardPoint(x: 15, y: 9), BoardPoint(x: 15, y: 15)], dimensions: dimensions)
            } else if boardWidth == 13 && boardHeight == 13 {
                // Draw star points for 13x13 board
                drawStarPointsForSize(points: [BoardPoint(x: 6, y: 6), BoardPoint(x: 3, y: 3), BoardPoint(x: 3, y: 9), BoardPoint(x: 9, y: 3), BoardPoint(x: 9, y: 9)], dimensions: dimensions)
            } else if boardWidth == 9 && boardHeight == 9 {
                // Draw star points for 9x9 board
                drawStarPointsForSize(points: [BoardPoint(x: 4, y: 4), BoardPoint(x: 2, y: 2), BoardPoint(x: 2, y: 6), BoardPoint(x: 6, y: 2), BoardPoint(x: 6, y: 6)], dimensions: dimensions)
            }
        }
    }
}

struct BoardLineView_Previews: PreviewProvider {
    static let board = ObservableBoard()
    static var previews: some View {
        GeometryReader { geometry in
            let dimensions = Dimensions(geometry: geometry, board: board)
            BoardLineView(dimensions: dimensions, boardWidth: board.width, boardHeight: board.height)
        }
        .onAppear() {
            BoardLineView_Previews.board.width = 13
            BoardLineView_Previews.board.height = 13
        }
    }
}
