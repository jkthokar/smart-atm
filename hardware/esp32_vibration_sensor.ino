// Optional hardware tamper sensor for Smart ATM.
// ESP32 + SW-420 vibration sensor. When the ATM is struck, the ESP32 sends
// POST /tamper to the computer running `python run.py`.
//
// Wiring: SW-420 VCC -> 3V3, GND -> GND, DO -> GPIO 27
// Set WIFI_SSID, WIFI_PASS and SERVER_URL below, then upload with the Arduino IDE.

#include <WiFi.h>
#include <HTTPClient.h>

const char* WIFI_SSID  = "your-wifi";
const char* WIFI_PASS  = "your-password";
const char* SERVER_URL = "http://192.168.0.10:8765/tamper";  // IP of the Smart ATM computer

const int SENSOR_PIN = 27;
const int HITS_TO_TRIGGER = 3;          // vibration pulses needed within the window
const unsigned long WINDOW_MS = 500;    // time window for counting pulses
const unsigned long COOLDOWN_MS = 5000; // minimum time between alarms

unsigned long windowStart = 0, lastAlarm = 0;
int hits = 0;

void setup() {
  Serial.begin(115200);
  pinMode(SENSOR_PIN, INPUT);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.printf("\nConnected, IP %s\n", WiFi.localIP().toString().c_str());
}

void sendAlarm() {
  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST("{\"sensor\":\"esp32-sw420\"}");
  Serial.printf("Alarm sent, HTTP %d\n", code);
  http.end();
}

void loop() {
  unsigned long now = millis();
  if (digitalRead(SENSOR_PIN) == HIGH) {
    if (now - windowStart > WINDOW_MS) { windowStart = now; hits = 0; }
    hits++;
    delay(20);  // debounce
  }
  if (hits >= HITS_TO_TRIGGER && now - lastAlarm > COOLDOWN_MS && WiFi.status() == WL_CONNECTED) {
    sendAlarm();
    lastAlarm = now;
    hits = 0;
  }
}
