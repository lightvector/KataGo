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

struct BoardPoint: Hashable {
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

struct ContentView: View {
    @StateObject var stones: Stones = Stones()
    @StateObject var messagesObject: MessagesObject = MessagesObject()
    @StateObject var board: Board = Board()
    @State private var selection: Tab = .command
    @State private var isShowingBoard: Bool = false
    @State private var boardText: [String] = []

    enum Tab {
        case command
        case goban
    }

    init() {
        // Start a thread to run KataGo GTP
        Thread {
            KataGoHelper.runGtp()
        }.start()
    }

    var body: some View {
        TabView(selection: $selection) {
            CommandView()
                .tabItem {
                    Label("Command", systemImage: "text.alignleft")
                }
                .tag(Tab.command)

            GobanView()
                .tabItem {
                    Label("Goban", systemImage: "circle")
                }
                .tag(Tab.goban)
        }
        .environmentObject(stones)
        .environmentObject(messagesObject)
        .environmentObject(board)
        .onAppear() {
            // Get messages from KataGo and append to the list of messages
            createMessageTask()
        }
    }

    /// Create message task
    private func createMessageTask() {
        Task {
            messagesObject.messages.append(Message(text: "Initializing..."))
            KataGoHelper.sendCommand("showboard")
            while true {
                let line = await Task.detached {
                    // Get a message line from KataGo
                    return KataGoHelper.getMessageLine()
                }.value

                // Create a message with the line
                let message = Message(text: line)

                // Append the message to the list of messages
                messagesObject.messages.append(message)

                // Collect board information
                maybeCollectBoard(message: line)
            }
        }
    }

    func maybeCollectBoard(message: String) {
        if isShowingBoard {
            if message.prefix(11) == "Next player" {
                isShowingBoard = false
                (stones.blackPoints, stones.whitePoints, board.width, board.height) = parseBoardPoints(board: boardText)
            } else {
                boardText.append(message)
            }
        } else {
            if message.prefix(9) == "= MoveNum" {
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
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
