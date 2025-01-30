//
//  STTTool.swift
//  stt-tool
//
//  Created by Zane Shannon on 1/15/25.
//

import ArgumentParser
import Foundation
import MLX
import MLXSTT

@main
struct STTTool: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Command line tool for converting speech to text",
        subcommands: [EvaluateCommand.self],
        defaultSubcommand: EvaluateCommand.self
    )
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

/// Command line arguments for transcribing audio.
struct TranscribeArguments: ParsableArguments {
    @Option(name: .long, help: "The name of transcription/translation output files before output format extensions")
    var outputName: String?

    @Option(name: [.customShort("o"), .long], help: "Directory to save the outputs")
    var outputDir = "."

    @Option(name: [.customShort("f"), .long], help: "Format of the output file")
    var outputFormat: OutputFormat = .txt

    @Flag(name: .long, inversion: .prefixedNo, help: "Whether to print out progress and debug messages")
    var verbose = true

    @Option(name: .long, help: "Perform speech recognition ('transcribe') or speech translation ('translate')")
    var task: Task = .transcribe

    @Option(name: .long, help: "Language spoken in the audio, specify nil to auto-detect")
    var language: String?

    @Option(name: .long, help: "Temperature for sampling")
    var temperature: Float = 0

    @Option(name: .long, help: "Number of candidates when sampling with non-zero temperature")
    var bestOf: Int = 5

    @Option(name: .long, help: "Optional patience value to use in beam decoding")
    var patience: Float?

    @Option(name: .long, help: "Optional token length penalty coefficient (alpha)")
    var lengthPenalty: Float?

    @Option(name: .long, help: "Comma-separated list of token ids to suppress during sampling")
    var suppressTokens = "-1"

    @Option(name: .long, help: "Optional text to provide as a prompt for the first window")
    var initialPrompt: String?

    @Flag(name: .long, inversion: .prefixedNo, help: "Whether to condition on previous text")
    var conditionOnPreviousText = true

    @Flag(name: .long, inversion: .prefixedNo, help: "Whether to perform inference in fp16")
    var fp16 = true

    @Option(name: .long, help: "Compression ratio threshold for failed decoding")
    var compressionRatioThreshold: Float = 2.4

    @Option(name: .long, help: "Log probability threshold for failed decoding")
    var logprobThreshold: Float = -1.0

    @Option(name: .long, help: "No speech threshold for silence detection")
    var noSpeechThreshold: Float = 0.6

    @Flag(name: .long, help: "Extract word-level timestamps")
    var wordTimestamps = false

    @Option(name: .long, help: "Punctuation symbols to merge with the next word")
    var prependPunctuations = "\"'“¿([{-"

    @Option(name: .long, help: "Punctuation symbols to merge with the previous word")
    var appendPunctuations = "\"'.。,，!！?？:：”)]}、"

    enum OutputFormat: String, ExpressibleByArgument {
        case txt, vtt, srt, tsv, json, all
    }

    enum Task: String, ExpressibleByArgument {
        case transcribe, translate
    }

    func transcribe(input: STTMInput, context: ModelContext) throws -> GenerateResult {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

        return try generate(
            input: input, parameters: .init(), context: context
        ) { tokens in

            if let last = tokens.last {
                detokenizer.append(token: last)
            }

            if let new = detokenizer.next() {
                print(new, terminator: "")
                fflush(stdout)
            }

//            if tokens.count >= maxTokens {
//                return .stop
//            } else {
            return .more
//            }
        }
    }
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

struct EvaluateCommand: AsyncParsableCommand {
    @OptionGroup var args: ModelArguments
    @OptionGroup var memory: MemoryArguments
    @OptionGroup var transcribe: TranscribeArguments

    @Option(name: .shortAndLong, help: "The audio file to transcribe") var audio: [URL]

    private func userInput(modelConfiguration _: ModelConfiguration) -> UserInput {
        var input = UserInput(audio: audio)
        return input
    }

    @MainActor
    mutating func run() async throws {
        if transcribe.verbose {
            print(args)
            print(memory)
            print(transcribe)
        }

        let writer = try getWriter(outputFormat: transcribe.outputFormat.rawValue, outputDir: transcribe.outputDir)
        // load the model
        let modelContainer = try await memory.start { [args] in
            try await args.load(defaultModel: "mlx-community/whisper-tiny", modelFactory: STTModelFactory.shared)
        }

        let userInput = self.userInput(modelConfiguration: modelContainer.configuration)

        let result = try await modelContainer.perform { context in
            let input = try await context.processor.prepare(input: userInput)
            return try transcribe.transcribe(input: input, context: context)
        }
        print("result", result)
        // TODO: writer.write(result)
        try print("Hello, world!", Data(contentsOf: audio.first!).count)
    }
}
