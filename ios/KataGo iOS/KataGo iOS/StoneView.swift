//
//  StoneView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/6.
//

import SwiftUI

struct StoneView: View {
    @EnvironmentObject var stones: Stones
    @EnvironmentObject var board: ObservableBoard
    let geometry: GeometryProxy

    var body: some View {
        let dimensions = Dimensions(geometry: geometry, board: board)
        drawStones(dimensions: dimensions)
    }

    private func drawStoneBase(stoneColor: Color, x: Int, y: Int, dimensions: Dimensions) -> some View {
        Circle()
            .foregroundColor(stoneColor)
            .frame(width: dimensions.squareLength, height: dimensions.squareLength)
            .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                      y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
    }

    private func drawLightEffect(stoneColor: Color, x: Int, y: Int, dimensions: Dimensions) -> some View {
        Circle()
            .fill(RadialGradient(gradient: Gradient(colors: [stoneColor, Color.white, Color.white]), center: .center, startRadius: dimensions.squareLength / 4, endRadius: 0))
            .offset(x: -dimensions.squareLength / 8, y: -dimensions.squareLength / 8)
            .padding(dimensions.squareLength / 4)
            .frame(width: dimensions.squareLength, height: dimensions.squareLength)
            .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                      y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
            .overlay {
                // Mask some light
                Circle()
                    .foregroundColor(stoneColor)
                    .blur(radius: dimensions.squareLength / 16)
                    .frame(width: dimensions.squareLength / 2, height: dimensions.squareLength / 2)
                    .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                              y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
            }
    }

    private func drawBlackStone(x: Int, y: Int, dimensions: Dimensions) -> some View {

        ZStack {
            // Black stone
            drawStoneBase(stoneColor: .black, x: x, y: y, dimensions: dimensions)

            // Light source effect
            drawLightEffect(stoneColor: .black, x: x, y: y, dimensions: dimensions)
        }
    }

    private func drawBlackStones(dimensions: Dimensions) -> some View {
        Group {
            ForEach(stones.blackPoints, id: \.self) { point in
                drawBlackStone(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawWhiteStone(x: Int, y: Int, dimensions: Dimensions) -> some View {

        ZStack {
            // Make a white stone darker than light
            let stoneColor = Color(white: 0.9)

            // White stone
            drawStoneBase(stoneColor: stoneColor, x: x, y: y, dimensions: dimensions)

            // Light source effect
            drawLightEffect(stoneColor: stoneColor, x: x, y: y, dimensions: dimensions)
        }
    }

    private func drawWhiteStones(dimensions: Dimensions) -> some View {
        Group {
            ForEach(stones.whitePoints, id: \.self) { point in
                drawWhiteStone(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawShadow(x: Int, y: Int, dimensions: Dimensions) -> some View {
        Group {
            // Shifted shadow
            Circle()
                .shadow(radius: dimensions.squareLength / 16, x: dimensions.squareLength / 8, y: dimensions.squareLength / 8)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)

            // Centered shadow
            Circle()
                .stroke(Color.black.opacity(0.5), lineWidth: dimensions.squareLength / 16)
                .blur(radius: dimensions.squareLength / 16)
                .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                .position(x: dimensions.marginWidth + CGFloat(x) * dimensions.squareLength,
                          y: dimensions.marginHeight + CGFloat(y) * dimensions.squareLength)
        }
    }

    private func drawShadows(dimensions: Dimensions) -> some View {
        Group {
            ForEach(stones.blackPoints, id: \.self) { point in
                drawShadow(x: point.x, y: point.y, dimensions: dimensions)
            }

            ForEach(stones.whitePoints, id: \.self) { point in
                drawShadow(x: point.x, y: point.y, dimensions: dimensions)
            }
        }
    }

    private func drawStones(dimensions: Dimensions) -> some View {
        ZStack {
            drawShadows(dimensions: dimensions)

            Group {
                drawBlackStones(dimensions: dimensions)
                drawWhiteStones(dimensions: dimensions)
            }
        }
    }
}

struct StoneView_Previews: PreviewProvider {
    static let stones = Stones()
    static let board = ObservableBoard()
    static var previews: some View {
        ZStack {
            Rectangle()
                .foregroundColor(.brown)

            GeometryReader { geometry in
                StoneView(geometry: geometry)
            }
            .environmentObject(stones)
            .environmentObject(board)
            .onAppear() {
                StoneView_Previews.board.width = 2
                StoneView_Previews.board.height = 2
                StoneView_Previews.stones.blackPoints = [BoardPoint(x: 0, y: 0), BoardPoint(x: 1, y: 1)]
                StoneView_Previews.stones.whitePoints = [BoardPoint(x: 0, y: 1), BoardPoint(x: 1, y: 0)]
            }
        }
    }
}
