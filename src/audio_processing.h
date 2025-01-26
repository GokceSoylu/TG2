#ifndef AUDIO_PROCESSING_H
#define AUDIO_PROCESSING_H

#include <Arduino.h>

void setupAudioProcessing();
String detectCommand();
void playAudioFeedback(const char *message);

#endif
