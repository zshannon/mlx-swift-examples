import MLXWhisper
import XCTest

final class TranscribeTests: XCTestCase {
    func testTranscribeJFK() async throws {
        let url: URL
        if let local = Bundle.module.url(forResource: "jfk", withExtension: "wav") {
            url = local
        } else {
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("jfk.wav")
            if !FileManager.default.fileExists(atPath: tmp.path) {
                let remote = URL(string: "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav")!
                let data = try Data(contentsOf: remote)
                try data.write(to: tmp)
            }
            url = tmp
        }
        let container = try await WhisperModelFactory.shared.loadContainer(
            configuration: WhisperRegistry.tiny)
        let text = try await container.transcribe(file: url.path)
        XCTAssertTrue(text.lowercased().contains("ask not"))
    }
}
