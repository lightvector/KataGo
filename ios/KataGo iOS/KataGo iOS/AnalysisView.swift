//
//  AnalysisView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/9/7.
//

import SwiftUI

struct AnalysisView: View {
    @EnvironmentObject var analysis: Analysis
    @EnvironmentObject var board: ObservableBoard
    let geometry: GeometryProxy

    var dimensions: Dimensions {
        Dimensions(geometry: geometry, board: board)
    }

    var shadows: some View {
        ForEach(analysis.data, id: \.self) { data in
            if let move = data["move"] {
                if let point = moveToPoint(move: move) {
                    // Shadow
                    Circle()
                        .stroke(Color.black.opacity(0.5), lineWidth: dimensions.squareLength / 32)
                        .blur(radius: dimensions.squareLength / 32)
                        .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                        .position(x: dimensions.marginWidth + CGFloat(point.x) * dimensions.squareLength,
                                  y: dimensions.marginHeight + CGFloat(point.y) * dimensions.squareLength)
                }
            }
        }
    }

    func computeDefiniteness(_ whiteness: Double) -> Double {
        return Swift.abs(whiteness - 0.5) * 2
    }

    var ownerships: some View {
        let sortedOwnershipKeys = analysis.ownership.keys.sorted()

        return ForEach(sortedOwnershipKeys, id: \.self) { point in
            if let ownership = analysis.ownership[point] {
                let whiteness = (analysis.nextColorForAnalysis == .white) ? (Double(ownership.mean) + 1) / 2 : (Double(-ownership.mean) + 1) / 2
                let definiteness = computeDefiniteness(whiteness)
                // Show a black or white square if definiteness is high and stdev is low
                // Show nothing if definiteness is low and stdev is low
                // Show a square with linear gradient of black and white if definiteness is low and stdev is high
                let scale = max(CGFloat(definiteness), CGFloat(ownership.stdev ?? 0)) * 0.7

                Rectangle()
                    .foregroundColor(Color(hue: 0, saturation: 0, brightness: whiteness).opacity(0.8))
                    .frame(width: dimensions.squareLength * scale, height: dimensions.squareLength * scale)
                    .position(x: dimensions.marginWidth + CGFloat(point.x) * dimensions.squareLength,
                              y: dimensions.marginHeight + CGFloat(point.y) * dimensions.squareLength)
            }
        }
    }

    var moves: some View {
        let maxVisits = computeMaxVisits()

        return ForEach(analysis.data, id: \.self) { data in
            if let move = data["move"] {
                if let point = moveToPoint(move: move) {
                    let winrate = Float(data["winrate"] ?? "0") ?? 0
                    let visits = Int(data["visits"] ?? "0") ?? 0
                    let isHidden = Float(visits) < (0.1 * Float(maxVisits))
                    let color = computeColorByVisits(isHidden: isHidden, visits: visits, maxVisits: maxVisits)

                    ZStack {
                        Circle()
                            .foregroundColor(color)
                        if !isHidden {
                            VStack {
                                Text(String(format: "%2.0f%%", winrate * 100))
                                    .font(.system(size: 500))
                                    .minimumScaleFactor(0.01)
                                    .bold()

                                Text(convertToSIUnits(visits))
                                    .font(.system(size: 500))
                                    .minimumScaleFactor(0.01)

                                if let scoreLead = data["scoreLead"] {
                                    let text = String(format: "%+.1f", (Float(scoreLead) ?? 0))
                                    Text(text)
                                        .font(.system(size: 500))
                                        .minimumScaleFactor(0.01)
                                }
                            }
                        }
                    }
                    .frame(width: dimensions.squareLength, height: dimensions.squareLength)
                    .position(x: dimensions.marginWidth + CGFloat(point.x) * dimensions.squareLength,
                              y: dimensions.marginHeight + CGFloat(point.y) * dimensions.squareLength)
                }
            }
        }
    }

    var body: some View {
        shadows
        ownerships
        moves
    }

    func convertToSIUnits(_ number: Int) -> String {
        let prefixes: [(prefix: String, value: Int)] = [
            ("T", 1_000_000_000_000),   // Tera
            ("G", 1_000_000_000),      // Giga
            ("M", 1_000_000),          // Mega
            ("k", 1_000)               // Kilo
        ]

        var result = Double(number)

        for (prefix, threshold) in prefixes {
            if number >= threshold {
                result = Double(number) / Double(threshold)
                return String(format: "%.1f%@", result, prefix)
            }
        }

        return "\(number)"
    }

    func computeColorByWinrate(isHidden: Bool, winrate: Float, minWinrate: Float, maxWinrate: Float) -> Color {
        let opacity = isHidden ? 0.1 : 0.5

        if winrate == maxWinrate {
            return .cyan.opacity(opacity)
        } else {
            let ratio = min(1, max(0.01, winrate - minWinrate) / max(0.01, maxWinrate - minWinrate))

            let fraction = 2 / (pow((1 / ratio) - 1, 0.9) + 1)

            if fraction < 1 {
                let hue = cbrt(fraction * fraction) / 2
                return Color(hue: Double(hue) / 2, saturation: 1, brightness: 1).opacity(opacity)
            } else {
                let hue = 1 - (sqrt(2 - fraction) / 2)
                return Color(hue: Double(hue) / 2, saturation: 1, brightness: 1).opacity(opacity)
            }
        }
    }

    func computeBaseColorByVisits(visits: Int, maxVisits: Int) -> Color {
        if visits == maxVisits {
            return Color(red: 0, green: 1, blue: 1)
        } else {
            let ratio = min(1, max(0.01, Float(visits)) / max(0.01, Float(maxVisits)))

            let fraction = 2 / (pow((1 / ratio) - 1, 0.9) + 1)

            if fraction < 1 {
                let hue = cbrt(fraction * fraction) / 2
                return Color(hue: Double(hue) / 2, saturation: 1, brightness: 1)
            } else {
                let hue = 1 - (sqrt(2 - fraction) / 2)
                return Color(hue: Double(hue) / 2, saturation: 1, brightness: 1)
            }
        }
    }

    func computeColorByVisits(isHidden: Bool, visits: Int, maxVisits: Int) -> Color {
        let baseColor = computeBaseColorByVisits(visits: visits, maxVisits: maxVisits)
        let opacity = isHidden ? 0.2 : 0.8
        return baseColor.opacity(opacity)
    }

    func computeMinMaxWinrate() -> (Float, Float) {
        let winrates = analysis.data.map() { data in
            Float(data["winrate"] ?? "0") ?? 0
        }

        let minWinrate = winrates.reduce(1) {
            min($0, $1)
        }

        let maxWinrate = winrates.reduce(0) {
            max($0, $1)
        }

        return (minWinrate, maxWinrate)
    }

    func computeMaxVisits() -> Int {
        let allVisits = analysis.data.map() { data in
            Int(data["visits"] ?? "0") ?? 0
        }

        let maxVisits = allVisits.reduce(0) {
            max($0, $1)
        }

        return maxVisits
    }

    func moveToPoint(move: String) -> BoardPoint? {
        // Mapping letters A-AD (without I) to numbers 0-28
        let letterMap: [String: Int] = [
            "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
            "F": 5, "G": 6, "H": 7, "J": 8, "K": 9,
            "L": 10, "M": 11, "N": 12, "O": 13, "P": 14,
            "Q": 15, "R": 16, "S": 17, "T": 18, "U": 19,
            "V": 20, "W": 21, "X": 22, "Y": 23, "Z": 24,
            "AA": 25, "AB": 26, "AC": 27, "AD": 28
        ]

        let pattern = /([^\d\W]+)(\d+)/
        if let match = move.firstMatch(of: pattern) {
            if let x = letterMap[String(match.1).uppercased()],
               let y = Int(match.2) {
                // Subtract 1 from y to make it 0-indexed
                return BoardPoint(x: x, y: y - 1)
            } else {
                return nil
            }
        } else {
            return nil
        }
    }
}

struct AnalysisView_Previews: PreviewProvider {
    static let analysis = Analysis()
    static let board = ObservableBoard()

    static var previews: some View {
        ZStack {
            Rectangle()
                .foregroundColor(.brown)

            GeometryReader { geometry in
                AnalysisView(geometry: geometry)
            }
            .environmentObject(analysis)
            .environmentObject(board)
            .onAppear() {
                AnalysisView_Previews.board.width = 2
                AnalysisView_Previews.board.height = 2
                AnalysisView_Previews.analysis.data = [["move": "A1", "winrate": "0.54321012345", "scoreLead": "0.123456789", "order": "0", "visits": "12345678"], ["move": "B1", "winrate": "0.4", "scoreLead": "-9.8", "order": "1", "visits": "2345678"], ["move": "A2", "winrate": "0.321", "scoreLead": "-12.345", "order": "2", "visits": "198"]]
                AnalysisView_Previews.analysis.ownership = [BoardPoint(x: 0, y: 0): Ownership(mean: 0.12, stdev: 0.5), BoardPoint(x: 1, y: 0): Ownership(mean: 0.987654321, stdev: 0.1), BoardPoint(x: 0, y: 1): Ownership(mean: -0.123456789, stdev: 0.4), BoardPoint(x: 1, y: 1): Ownership(mean: -0.98, stdev: 0.2)]
            }
        }
    }
}
