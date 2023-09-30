//
//  ContentView.swift
//  KataGo iOS
//
//  Created by Chin-Chang Yang on 2023/7/2.
//

import SwiftUI

class Board: ObservableObject {
    @Published var width: CGFloat = 19
    @Published var height: CGFloat = 19
}

struct BoardPoint: Hashable, Comparable {
    static func < (lhs: BoardPoint, rhs: BoardPoint) -> Bool {
        if lhs.y > rhs.y {
            return false
        } else if lhs.y < rhs.y {
            return true
        } else if lhs.x < rhs.x {
            return true
        } else {
            return false
        }
    }

    let x: Int
    let y: Int
}

class Stones: ObservableObject {
    @Published var blackPoints: [BoardPoint] = []
    @Published var whitePoints: [BoardPoint] = []
}

class MessagesObject: ObservableObject {
    @Published var messages: [Message] = []
}

enum PlayerColor {
    case black
    case white
}

class PlayerObject: ObservableObject {
    @Published var nextPlay = PlayerColor.black
    @Published var nextShow = PlayerColor.black
}

struct Ownership {
    let mean: Float
    let stdev: Float?

    init(mean: Float, stdev: Float?) {
        self.mean = mean
        self.stdev = stdev
    }

    init(mean: Float) {
        self.init(mean: mean, stdev: nil)
    }
}

class Analysis: ObservableObject {
    @Published var nextShow = PlayerColor.white
    @Published var data: [[String: String]] = []
    @Published var ownership: [BoardPoint: Ownership] = [:]
}

class Config: ObservableObject {
    static let defaultMaxMessageCharacters: Int = 200
    static let defaultMaxAnalysisMoves: Int = 8
    @Published var maxMessageCharacters: Int = defaultMaxMessageCharacters
    @Published var maxAnalysisMoves: Int = defaultMaxAnalysisMoves
}

struct ContentView: View {
    @StateObject var stones = Stones()
    @StateObject var messagesObject = MessagesObject()
    @StateObject var board = Board()
    @StateObject var player = PlayerObject()
    @StateObject var analysis = Analysis()
    @StateObject var config = Config()
    @State private var isShowingBoard = false
    @State private var boardText: [String] = []

    init() {
        // Start a thread to run KataGo GTP
        Thread {
            KataGoHelper.runGtp()
        }.start()
    }

    var body: some View {
        TabView() {
            CommandView()
                .tabItem {
                    Label("Command", systemImage: "text.alignleft")
                }

            GobanView()
                .tabItem {
                    Label("Goban", systemImage: "circle")
                }

            ConfigView()
                .tabItem {
                    Label("Config", systemImage: "slider.horizontal.3")
                }
        }
        .environmentObject(stones)
        .environmentObject(messagesObject)
        .environmentObject(board)
        .environmentObject(player)
        .environmentObject(analysis)
        .environmentObject(config)
        .onAppear() {
            // Get messages from KataGo and append to the list of messages
            createMessageTask()
        }
    }

    /// Create message task
    private func createMessageTask() {
        Task {
            messagesObject.messages.append(Message(text: "Initializing...", maxLength: config.maxMessageCharacters))
            KataGoHelper.sendCommand("showboard")
            while true {
                let line = await Task.detached {
                    // Get a message line from KataGo
                    return KataGoHelper.getMessageLine()
                }.value

                // Create a message with the line
                let message = Message(text: line, maxLength: config.maxMessageCharacters)

                // Append the message to the list of messages
                messagesObject.messages.append(message)

                // Collect board information
                maybeCollectBoard(message: line)

                // Collect analysis information
                maybeCollectAnalysis(message: line)

                // Remove when there are too many messages
                while messagesObject.messages.count > 100 {
                    messagesObject.messages.removeFirst()
                }
            }
        }
    }

    func maybeCollectBoard(message: String) {
        if isShowingBoard {
            if message.prefix("Next player".count) == "Next player" {
                isShowingBoard = false
                (stones.blackPoints, stones.whitePoints, board.width, board.height) = parseBoardPoints(board: boardText)
                if message.prefix("Next player: Black".count) == "Next player: Black" {
                    player.nextPlay = .black
                    player.nextShow = .black
                } else {
                    player.nextPlay = .white
                    player.nextShow = .white
                }
            } else {
                boardText.append(message)
            }
        } else {
            if message.prefix("= MoveNum".count) == "= MoveNum" {
                boardText = []
                isShowingBoard = true
            }
        }
    }

    func parseBoardPoints(board: [String]) -> ([BoardPoint], [BoardPoint], CGFloat, CGFloat) {
        var blackStones: [BoardPoint] = []
        var whiteStones: [BoardPoint] = []

        let height = CGFloat(board.count - 1)  // Subtracting 1 to exclude the header
        let width = CGFloat((board.last?.dropFirst(2).count ?? 0) / 2)  // Drop the first 2 characters for the y-coordinate and divide by 2 because of spaces between cells

        // Start from index 1 to skip the header line
        for (lineIndex, line) in board.enumerated() where lineIndex > 0 {
            // Get y-coordinate from the beginning of the line, and subtract 1 to start from 0
            let y = (Int(line.prefix(2).trimmingCharacters(in: .whitespaces)) ?? 1) - 1

            // Start parsing after the space that follows the y-coordinate
            for (charIndex, char) in line.dropFirst(3).enumerated() where char == "X" || char == "O" {
                let xCoord = charIndex / 2
                if char == "X" {
                    blackStones.append(BoardPoint(x: xCoord, y: y))
                } else if char == "O" {
                    whiteStones.append(BoardPoint(x: xCoord, y: y))
                }
            }
        }

        return (blackStones, whiteStones, width, height)
    }

    func maybeCollectAnalysis(message: String) {
        if message.starts(with: /info/) {
            let splitData = message.split(separator: "info")
            analysis.data = splitData.map {
                extractMoveData(dataLine: String($0))
            }

            if let lastData = splitData.last {
                analysis.ownership = extractOwnership(message: String(lastData))
            }

            analysis.nextShow = player.nextShow
        }
    }

    func extractMoveData(dataLine: String) -> [String: String] {
        // Define patterns for extracting relevant information
        let patterns: [String: Regex] = [
            "move": /move (\w\d+)/,
            "visits": /visits (\d+)/,
            "winrate": /winrate ([\d.eE]+)/,
            "scoreLead": /scoreLead ([-\d.eE]+)/
        ]

        var moveData: [String: String] = [:]
        for (key, pattern) in patterns {
            if let match = dataLine.firstMatch(of: pattern) {
                moveData[key] = String(match.1)
            }
        }

        return moveData
    }

    func extractOwnershipMean(message: String) -> [Float] {
        let pattern = /ownership ([-\d\s.eE]+)/
        if let match = message.firstMatch(of: pattern) {
            let mean = match.1.split(separator: " ").compactMap { Float($0)
            }
            assert(mean.count == Int(board.width * board.height))
            return mean
        }

        return []
    }

    func extractOwnershipStdev(message: String) -> [Float] {
        let pattern = /ownershipStdev ([-\d\s.eE]+)/
        if let match = message.firstMatch(of: pattern) {
            let stdev = match.1.split(separator: " ").compactMap { Float($0)
            }
            assert(stdev.count == Int(board.width * board.height))
            return stdev
        }

        return []
    }

    func extractOwnership(message: String) -> [BoardPoint: Ownership] {
        let mean = extractOwnershipMean(message: message)
        let stdev = extractOwnershipStdev(message: message)
        if !mean.isEmpty && !stdev.isEmpty {
            var dictionary: [BoardPoint: Ownership] = [:]
            var i = 0
            for y in stride(from:Int(board.height - 1), through: 0, by: -1) {
                for x in 0..<Int(board.width) {
                    let point = BoardPoint(x: x, y: y)
                    dictionary[point] = Ownership(mean: mean[i], stdev: stdev[i])
                    i = i + 1
                }
            }
            return dictionary
        }

        return [:]
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
