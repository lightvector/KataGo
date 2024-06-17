import Foundation

class StandardError: TextOutputStream {
    /// A shared instance of the StandardError class.
    static var instance = StandardError()

    /// Writes the given string to standard error output.
    func write(_ string: String) {
        /// Attempts to write the contents of a Data object containing the UTF8-encoded string to
        /// the standard error file handle.
        try? FileHandle.standardError.write(contentsOf: Data(string.utf8))
    }
}

func printError(_ item: Any) {
    print(item, to: &StandardError.instance)
}
