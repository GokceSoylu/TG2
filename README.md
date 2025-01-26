# voice_command_project
ben yoruldum hayat 🥲

voice_command_project/
├── main.ino                // Ana kod dosyası
├── commands.h              // Komut listesi ve ilgili tanımlar
├── audio_processing.cpp    // Ses işleme ve tanıma
├── audio_processing.h      // Ses işleme başlık dosyası
├── lib/                    // Ek kütüphaneler
│   ├── TensorFlowLite/
│   └── I2SDriver/
└── README.md               // Proje açıklamaları



# Voice Command Recognition for Home Automation

## Proje Özeti
Bu proje, sesli komutları tanıyıp ev otomasyonu cihazı tasarlamak için geliştirilmiştir. Projede ESP32, I2S mikrofon modülü ve hoparlör kullanılmaktadır.

## Komutlar
1. Turn on the light
2. Turn off the light
3. Increase volume
4. Decrease volume
5. Play music
TURN_ON_LIGHT için: on
TURN_OFF_LIGHT için:  off
INCREASE_VOLUME için: up
DECREASE_VOLUME için: down
PLAY_MUSIC için: go 
## Donanım Bağlantıları
- INMP441:
  - VDD → 3.3V
  - GND → GND
  - SD → GPIO32
  - SCK → GPIO14
  - WS → GPIO15

- MAX98357:
  - VIN → 5V
  - GND → GND
  - DIN → GPIO22
  - LRC → GPIO25
  - BCLK → GPIO26
