//
//  Writers.swift
//  mlx-swift-examples
//
//  Created by Zane Shannon on 1/29/25.
//


import Foundation

// MARK: - Timestamp Formatting

func formatTimestamp(seconds: Double, alwaysIncludeHours: Bool = false, decimalMarker: String = ".") -> String {
    precondition(seconds >= 0, "non-negative timestamp expected")
    var milliseconds = Int(round(seconds * 1000.0))
    
    let hours = milliseconds / 3_600_000
    milliseconds -= hours * 3_600_000
    
    let minutes = milliseconds / 60_000
    milliseconds -= minutes * 60_000
    
    let secs = milliseconds / 1_000
    milliseconds -= secs * 1_000
    
    let hoursMarker = alwaysIncludeHours || hours > 0 ? String(format: "%02d:", hours) : ""
    return String(format: "%@%02d:%02d%@%03d", hoursMarker, minutes, secs, decimalMarker, milliseconds)
}

// MARK: - Helper Functions

func getStart(segments: [[String: Any]]) -> Double? {
    if let firstSegment = segments.first {
        if let words = firstSegment["words"] as? [[String: Any]] {
            return words.first?["start"] as? Double
        }
        return firstSegment["start"] as? Double
    }
    return nil
}

// MARK: - Base Protocol

public protocol ResultWriter {
    var `extension`: String { get }
    var outputDir: String { get }
    
    func writeResult(result: [String: Any], to file: FileHandle, options: [String: Any]?) throws
}

// MARK: - Text Writer

class WriteTXT: ResultWriter {
    let `extension` = "txt"
    let outputDir: String
    
    init(outputDir: String) {
        self.outputDir = outputDir
    }
    
    func writeResult(result: [String: Any], to file: FileHandle, options: [String: Any]?) throws {
        guard let segments = result["segments"] as? [[String: Any]] else { return }
        
        for segment in segments {
            if let text = segment["text"] as? String {
                let line = text.trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
                if let data = line.data(using: .utf8) {
                    file.write(data)
                }
            }
        }
    }
}

// MARK: - Subtitles Base Protocol

protocol SubtitlesWriter: ResultWriter {
    var alwaysIncludeHours: Bool { get }
    var decimalMarker: String { get }
}

extension SubtitlesWriter {
    func formatTimestamp(_ seconds: Double) -> String {
        MLXSTT.formatTimestamp(seconds: seconds, alwaysIncludeHours: alwaysIncludeHours, decimalMarker: decimalMarker)
    }
}

// MARK: - VTT Writer

class WriteVTT: SubtitlesWriter {
    let `extension` = "vtt"
    let outputDir: String
    let alwaysIncludeHours = false
    let decimalMarker = "."
    
    init(outputDir: String) {
        self.outputDir = outputDir
    }
    
    func writeResult(result: [String: Any], to file: FileHandle, options: [String: Any]?) throws {
        if let headerData = "WEBVTT\n\n".data(using: .utf8) {
            file.write(headerData)
        }
        
        guard let segments = result["segments"] as? [[String: Any]] else { return }
        
        for segment in segments {
            if let start = segment["start"] as? Double,
               let end = segment["end"] as? Double,
               let text = segment["text"] as? String {
                let startStr = formatTimestamp(start)
                let endStr = formatTimestamp(end)
                let line = "\(startStr) --> \(endStr)\n\(text.trimmingCharacters(in: .whitespacesAndNewlines))\n\n"
                if let data = line.data(using: .utf8) {
                    file.write(data)
                }
            }
        }
    }
}

// MARK: - Writer Factory

enum WriterError: Error {
    case unsupportedFormat
}

public func getWriter(outputFormat: String, outputDir: String) throws -> ResultWriter {
    switch outputFormat {
    case "txt":
        return WriteTXT(outputDir: outputDir)
    case "vtt":
        return WriteVTT(outputDir: outputDir)
    // Add other format handlers here
    default:
        throw WriterError.unsupportedFormat
    }
}
