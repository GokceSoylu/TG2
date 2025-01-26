#include "audio_processing.h"
#include <I2S.h>
#include <TensorFlowLite.h>

// Ses giriş ve çıkış pinleri
#define I2S_MIC_SD 32
#define I2S_MIC_SCK 14
#define I2S_MIC_WS 15

I2SSampler *i2sSampler;

void setupAudioProcessing()
{
    i2sSampler = new I2SSampler(I2S_MIC_SD, I2S_MIC_SCK, I2S_MIC_WS);
    i2sSampler->begin();
}

String detectCommand()
{
    int16_t *audioData = i2sSampler->readAudio();
    String recognizedCommand = processAudioData(audioData);
    return recognizedCommand;
}

void playAudioFeedback(const char *message)
{
    Serial.print("Audio Feedback: ");
    Serial.println(message);
    // Burada hoparlöre ses çıkışı sağlanabilir.
}
