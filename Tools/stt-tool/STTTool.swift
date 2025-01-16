//
//  STTTool.swift
//  stt-tool
//
//  Created by Zane Shannon on 1/15/25.
//

import ArgumentParser
import Foundation
import Hub
import MLX
import MLXSTT
import MLXLMCommon

@main
struct STTTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for converting speech to text",
        subcommands: [EvaluateCommand.self],
        defaultSubcommand: EvaluateCommand.self
    )
}

/// Argument package for adjusting and reporting memory use.
struct MemoryArguments: ParsableArguments, Sendable {
    @Flag(name: .long, help: "Show memory stats")
    var memoryStats = false

    @Option(name: .long, help: "Maximum cache size in M")
    var cacheSize: Int?

    @Option(name: .long, help: "Maximum memory size in M")
    var memorySize: Int?

    var startMemory: GPU.Snapshot?

    mutating func start<L>(_ load: @Sendable () async throws -> L) async throws -> L {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        let result = try await load()
        startMemory = GPU.snapshot()

        return result
    }

    mutating func start() {
        if let cacheSize {
            GPU.set(cacheLimit: cacheSize * 1024 * 1024)
        }

        if let memorySize {
            GPU.set(memoryLimit: memorySize * 1024 * 1024)
        }

        startMemory = GPU.snapshot()
    }

    func reportCurrent() {
        if memoryStats {
            let memory = GPU.snapshot()
            print(memory.description)
        }
    }

    func reportMemoryStatistics() {
        if memoryStats, let startMemory {
            let endMemory = GPU.snapshot()

            print("=======")
            print("Memory size: \(GPU.memoryLimit / 1024)K")
            print("Cache size:  \(GPU.cacheLimit / 1024)K")

            print("")
            print("=======")
            print("Starting memory")
            print(startMemory.description)

            print("")
            print("=======")
            print("Ending memory")
            print(endMemory.description)

            print("")
            print("=======")
            print("Growth")
            print(startMemory.delta(endMemory).description)
        }
    }
}

/// Command line arguments for loading a model.
struct ModelArguments: ParsableArguments, Sendable {
    @Option(name: .long, help: "Name of the huggingface model or absolute path to directory")
    var model: String?

    @Sendable
    func load(defaultModel: String, modelFactory: ModelFactory) async throws -> ModelContainer {
        let modelConfiguration: ModelConfiguration

        let modelName = model ?? defaultModel

        if modelName.hasPrefix("/") {
            // path
            modelConfiguration = ModelConfiguration(directory: URL(filePath: modelName))
        } else {
            // identifier
            modelConfiguration = modelFactory.configuration(id: modelName)
        }
        return try await modelFactory.loadContainer(configuration: modelConfiguration)
    }
}

struct EvaluateCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "eval",
        abstract: "evaluate prompt and generate text"
    )

    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments

    @Option(name: .shortAndLong, help: "The audio file to transcribe") var file: URL

    @MainActor
    mutating func run() async throws {
        let modelFactory: ModelFactory = STTModelFactory.shared
        let defaultModel: ModelConfiguration = .init(
            id: "mlx-community/whisper-tiny"
        )

        // load the model
        let modelContainer = try await memory.start { [args] in
            try await args.load(defaultModel: defaultModel.name, modelFactory: modelFactory)
        }
        try print("Hello, world!", Data(contentsOf: file).count)
    }
}
