#include <Arduino.h>
#include "commands.h"
#include "audio_processing.h"

// LED Pinleri
#define LED_PIN1 13
#define LED_PIN2 12
#define LED_PIN3 33

// put function declarations here:
int myFunction(int, int);

void setup()
{
  Serial.begin(115200);

  // LED pinlerini çıkış olarak ayarla
  pinMode(LED_PIN1, OUTPUT);
  pinMode(LED_PIN2, OUTPUT);
  pinMode(LED_PIN3, OUTPUT);

  // I2S ve ses işleme başlat
  setupAudioProcessing();
  Serial.println("System ready. Listening for commands...");
  // put your setup code here, to run once:
}

void loop()
{
  // Komut algılama
  String command = detectCommand();

  if (command == "TURN_ON_LIGHT")
  {
    digitalWrite(LED_PIN1, HIGH);
    Serial.println("Light turned ON");
  }
  else if (command == "TURN_OFF_LIGHT")
  {
    digitalWrite(LED_PIN1, LOW);
    Serial.println("Light turned OFF");
  }
  else if (command == "INCREASE_VOLUME")
  {
    digitalWrite(LED_PIN2, HIGH); // Sembol olarak kullanıyoruz
    delay(1000);
    digitalWrite(LED_PIN2, LOW);
    Serial.println("Volume increased");
  }
  else if (command == "DECREASE_VOLUME")
  {
    digitalWrite(LED_PIN3, HIGH); // Sembol olarak kullanıyoruz
    delay(1000);
    digitalWrite(LED_PIN3, LOW);
    Serial.println("Volume decreased");
  }
  else if (command == "PLAY_MUSIC")
  {
    playAudioFeedback("Playing music...");
  }
  // put your main code here, to run repeatedly:
}

// put function definitions here:
int myFunction(int x, int y)
{
  return x + y;
}