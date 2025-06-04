import MLXWhisper
import XCTest

final class TranscribeTests: XCTestCase {
    func testTranscribeJFK() async throws {
        guard let url = Bundle.module.url(forResource: "jfk", withExtension: "wav") else {
            XCTFail("Could not find jfk.wav resource in test bundle")
            return
        }
        
        let container = try await WhisperModelFactory.shared.loadContainer(
            configuration: WhisperRegistry.tiny)
        let text = try await container.transcribe(file: url.path)
        XCTAssertTrue(text.lowercased().contains("ask not"))
    }
}
