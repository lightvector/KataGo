//
//  GobanView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/2.
//

import SwiftUI

struct GobanView: View {
    @EnvironmentObject var stones: Stones

    let boardXLengh: CGFloat = 19
    let boardYLengh: CGFloat = 19
    let boardSpace: CGFloat = 20
    let texture = WoodImage.createTexture()

    var body: some View {
        GeometryReader { geometry in
            let dimensions = calculateBoardDimensions(geometry: geometry)
            ZStack {
                drawBoardBackground(texture: texture, dimensions: dimensions)
                drawLines(dimensions: dimensions)
                drawStarPoints(dimensions: dimensions)
                drawStones(dimensions: dimensions)
            }
        }
    }

    private func calculateBoardDimensions(geometry: GeometryProxy) -> (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat) {
        let totalWidth = geometry.size.width
        let totalHeight = geometry.size.height
        let squareWidth = (totalWidth - boardSpace) / boardXLengh
        let squareHeight = (totalHeight - boardSpace) / boardYLengh
        let squareLength = min(squareWidth, squareHeight)
        let boardWidth = boardXLengh * squareLength
        let boardHeight = boardYLengh * squareLength
        let marginWidth = (totalWidth - boardWidth + squareLength) / 2
        let marginHeight = (totalHeight - boardHeight + squareLength) / 2
        return (squareLength, boardWidth, boardHeight, marginWidth, marginHeight)
    }

    private func drawBoardBackground(texture: UIImage, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            Image(uiImage: texture)
                .resizable()
                .frame(width: dimensions.boardWidth, height: dimensions.boardHeight)
        }
    }

    private func drawLines(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            ForEach(0..<Int(boardYLengh), id: \.self) { i in
                horizontalLine(i: i, dimensions: dimensions)
            }
            ForEach(0..<Int(boardXLengh), id: \.self) { i in
                verticalLine(i: i, dimensions: dimensions)
            }
        }
    }

    private func horizontalLine(i: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Path { path in
            path.move(to: CGPoint(x: dimensions.marginWidth, y: dimensions.marginHeight + CGFloat(i) * dimensions.squareLength))
            path.addLine(to: CGPoint(x: dimensions.marginWidth + dimensions.boardWidth - dimensions.squareLength, y: dimensions.marginHeight + CGFloat(i) * dimensions.squareLength))
        }
        .stroke(Color.black)
    }

    private func verticalLine(i: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Path { path in
            path.move(to: CGPoint(x: dimensions.marginWidth + CGFloat(i) * dimensions.squareLength, y: dimensions.marginHeight))
            path.addLine(to: CGPoint(x: dimensions.marginWidth + CGFloat(i) * dimensions.squareLength, y: dimensions.marginHeight + dimensions.boardHeight - dimensions.squareLength))
        }
        .stroke(Color.black)
    }

    private func drawStarPoint(x: Int, y: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        // Big black dot
        Circle()
            .frame(width: dimensions.squareLength / 4, height: dimensions.squareLength / 4)
            .foregroundColor(Color.black)
            .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                      y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
    }

    private func drawStarPointsForSize(points: [BoardPoint], dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        ForEach(points, id: \.self) { point in
            drawStarPoint(x: point.x, y: point.y, dimensions: dimensions)
        }
    }

    private func drawStarPoints(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            if boardXLengh == 19 && boardYLengh == 19 {
                // Draw star points for 19x19 board
                drawStarPointsForSize(points: [BoardPoint(x: 3, y: 3), BoardPoint(x: 3, y: 9), BoardPoint(x: 3, y: 15), BoardPoint(x: 9, y: 3), BoardPoint(x: 9, y: 9), BoardPoint(x: 9, y: 15), BoardPoint(x: 15, y: 3), BoardPoint(x: 15, y: 9), BoardPoint(x: 15, y: 15)], dimensions: dimensions)
            } else if boardXLengh == 13 && boardYLengh == 13 {
                // Draw star points for 13x13 board
                drawStarPointsForSize(points: [BoardPoint(x: 6, y: 6), BoardPoint(x: 3, y: 3), BoardPoint(x: 3, y: 9), BoardPoint(x: 9, y: 3), BoardPoint(x: 9, y: 9)], dimensions: dimensions)
            } else if boardXLengh == 9 && boardYLengh == 9 {
                // Draw star points for 9x9 board
                drawStarPointsForSize(points: [BoardPoint(x: 4, y: 4), BoardPoint(x: 2, y: 2), BoardPoint(x: 2, y: 6), BoardPoint(x: 6, y: 2), BoardPoint(x: 6, y: 6)], dimensions: dimensions)
            }
        }
    }

    private func drawBlackStone(x: Int, y: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {

        ZStack {
            // Black stone
            Circle()
                .foregroundColor(.black)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Light source effect
            Circle()
                .fill(RadialGradient(gradient: Gradient(colors: [Color.black, Color.white]), center: .center, startRadius: dimensions.squareLength / 4, endRadius: 0))
                .offset(x: -dimensions.squareLength / 8, y: -dimensions.squareLength / 8)
                .padding(dimensions.squareLength / 4)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Mask some light
            Circle()
                .foregroundColor(.black)
                .blur(radius: dimensions.squareLength / 8)
                .frame(width: dimensions.squareLength / 2, height: dimensions.squareLength / 2)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
        }
    }

    private func drawBlackStones(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            ForEach(stones.blackPoints, id: \.self) { point in
                drawBlackStone(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawWhiteStone(x: Int, y: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {

        ZStack {
            // Make a white stone darker than light
            let stoneColor = Color(white: 0.85)

            // White stone
            Circle()
                .foregroundColor(stoneColor)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Light source effect
            Circle()
                .fill(RadialGradient(gradient: Gradient(colors: [stoneColor, Color.white]), center: .center, startRadius: dimensions.squareLength / 4, endRadius: 0))
                .offset(x: -dimensions.squareLength / 8, y: -dimensions.squareLength / 8)
                .padding(dimensions.squareLength / 4)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Mask some light
            Circle()
                .foregroundColor(stoneColor)
                .blur(radius: dimensions.squareLength / 8)
                .frame(width: dimensions.squareLength / 2, height: dimensions.squareLength / 2)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
        }
    }

    private func drawWhiteStones(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            ForEach(stones.whitePoints, id: \.self) { point in
                drawWhiteStone(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawShadow(x: Int, y: Int, dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            // Shifted shadow
            Circle()
                .shadow(radius: dimensions.squareLength / 16, x: dimensions.squareLength / 8, y: dimensions.squareLength / 8)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Centered shadow
            Circle()
                .shadow(radius: dimensions.squareLength / 8)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
        }
    }

    private func drawShadows(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        Group {
            ForEach(stones.blackPoints, id: \.self) { point in
                drawShadow(x: point.x, y: point.y, dimensions: dimensions)
            }

            ForEach(stones.whitePoints, id: \.self) { point in
                drawShadow(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawStones(dimensions: (squareLength: CGFloat, boardWidth: CGFloat, boardHeight: CGFloat, marginWidth: CGFloat, marginHeight: CGFloat)) -> some View {
        ZStack {
            drawShadows(dimensions: dimensions)

            Group {
                drawBlackStones(dimensions: dimensions)
                drawWhiteStones(dimensions: dimensions)
            }
        }
    }

}

struct GobanView_Previews: PreviewProvider {
    static let stones = Stones()

    static var previews: some View {
        GobanView()
            .environmentObject(stones)
            .onAppear() {
                GobanView_Previews.stones.blackPoints = [BoardPoint(x: 15, y: 3), BoardPoint(x: 13, y: 2), BoardPoint(x: 9, y: 3), BoardPoint(x: 3, y: 3)]
                GobanView_Previews.stones.whitePoints = [BoardPoint(x: 3, y: 15)]
            }
    }
}
