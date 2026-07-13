/*
 * canine_cooling_system.ino
 * ─────────────────────────────────────────────────────────────────────────
 * Canine Heat Stroke Prevention System — Blynk IoT Edition
 * Thesis: "IoT Monitoring of Vital Signs and Active Cooling for
 *          Canine Heat Stroke Prevention Using Binary Classification"
 *
 * ── Hardware ──────────────────────────────────────────────────────────────
 *   ESP32 development board
 *   DS18B20  waterproof probe  → body temperature  (GPIO4, OneWire)
 *   DHT22    sensor            → environmental temperature (GPIO5)
 *   Piezoelectric disc/film    → respiratory rate   (GPIO35 = ADC1_CH7)
 *     Piezo(+) ── GPIO35 ── [1 MΩ to GND]   (1 MΩ bias resistor to GND)
 *     Piezo(−) ── GND
 *   2-channel relay module     → Channel 1: Peltier module (GPIO25)
 *                                Channel 2: Cooling fans   (GPIO26)
 *
 * ── Required Arduino libraries (Library Manager) ──────────────────────────
 *   OneWire            by Paul Stoffregen
 *   DallasTemperature  by Miles Burton
 *   DHT sensor library by Adafruit
 *   Blynk              by Volodymyr Shymanskyy  (v1.3.x or newer)
 *
 * ── ML model header ───────────────────────────────────────────────────────
 *   Run  python export_model.py  → copy canine_model.h into this folder.
 *
 * ── Blynk dashboard virtual pins ─────────────────────────────────────────
 *   V0  Body temperature       (°C)   — Gauge, 36.5–42
 *   V1  Respiratory rate       (bpm)  — Gauge, 0–400
 *   V2  Environmental temp     (°C)   — Gauge, 15–45
 *   V4  ML result              (0/1)  — LED/Value, 0=Normal 1=HeatStressed
 *   V5  Cooling state          (0–3)  — Value, 0=NORMAL 1=STRESSED 2=COOL_P 3=COOL_F
 *
 * ── Data logging ──────────────────────────────────────────────────────────
 *   Every 60 s the ESP32 POSTs a JSON record to a Google Apps Script web
 *   app, which appends a row to a Google Sheet.
 *
 * ── Respiratory Rate Algorithm ────────────────────────────────────────────
 *   Covers full canine range: rest (10-29 bpm) → panting (200-400 bpm)
 *   Sample rate: 200 Hz (5 ms non-blocking millis gate)
 *
 *   FORMULA 1 — Bandpass Filter (two cascaded 1st-order IIR, Ts = 5 ms):
 *     LP[n]  = α_low  × x[n] + (1 − α_low)  × LP[n−1]   [low-pass]
 *     HP[n]  = α_high × x[n] + (1 − α_high) × HP[n−1]   [slow baseline]
 *     BP[n]  = LP[n] − HP[n]                              [bandpass]
 *     Cutoff: f = α / (2π × Ts)
 *       α_low  = 0.30 → f_high = 9.55 Hz = 573 bpm  (noise cut-off)
 *       α_high = 0.003 → f_low  = 0.095 Hz = 5.7 bpm (drift removal)
 *     → Passband 5.7–573 bpm covers rest (10 bpm) to panting (400 bpm)
 *
 *   FORMULA 2 — Adaptive Peak Envelope (threshold with no manual tuning):
 *     |BP[n]| > E[n−1]:  E[n] = α_atk × |BP[n]| + (1−α_atk) × E[n−1]
 *     |BP[n]| ≤ E[n−1]:  E[n] = α_dec × |BP[n]| + (1−α_dec) × E[n−1]
 *     τ_attack = Ts/α_atk = 5ms/0.10 =   50 ms
 *     τ_decay  = Ts/α_dec = 5ms/0.002 = 2500 ms
 *     Threshold  T[n] = max(T_min, E[n] × 0.50)
 *     Re-arm     R[n] = T[n] × 0.30
 *
 *   FORMULA 3 — Median BPM (immune to single outlier detections):
 *     gap_k = t_k − t_(k−1)              [ms between peaks]
 *     BPM_median = 60000 / median(gap_1…gap_5)
 *     BPM_out[n] = 0.15 × BPM_median + 0.85 × BPM_out[n−1]
 *
 *   Heat stress thresholds (thesis):
 *     NORMAL (rest)              10 – 29  bpm
 *     HEAT STRESSED (rest)       30 – 64  bpm
 *     NORMAL (walking/running)   65 – 199 bpm
 *     HEAT STRESSED (active)    200 – 400 bpm  [severe panting]
 *
 * ── Cooling state machine ─────────────────────────────────────────────────
 *   NORMAL ──(≥2 sensors stressed)──► HEAT_STRESSED
 *                                        │
 *                               (all readings return to normal)
 *                                        ▼
 *                                COOLDOWN_PELTIER  [60 s — Peltier+Fan ON]
 *                                        │  (60 s elapsed)
 *                                        ▼
 *                                COOLDOWN_FAN      [120 s — Fan only ON]
 *                                        │  (120 s elapsed)
 *                                        ▼
 *                                     NORMAL
 *   Any cooldown + stress re-detected → immediately back to HEAT_STRESSED.
 */

// ═══════════════════════════════════════════════════════════════════════════
//  BLYNK CREDENTIALS  —  must come before any #include
// ═══════════════════════════════════════════════════════════════════════════
#define BLYNK_TEMPLATE_ID    "TMPL6ON4E1cJ3"
#define BLYNK_TEMPLATE_NAME  "Canine Cooling System"
#define BLYNK_AUTH_TOKEN     "agAVZhjCKKq8kUqQr-3hBbVFFZn55hzh"
#define BLYNK_PRINT          Serial   // comment out after initial testing

// ═══════════════════════════════════════════════════════════════════════════
//  INCLUDES
// ═══════════════════════════════════════════════════════════════════════════
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <time.h>
#include <math.h>           // fabsf, fmaxf, roundf

#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include "canine_model.h"   // generated by export_model.py

// ═══════════════════════════════════════════════════════════════════════════
//  NETWORK CREDENTIALS
// ═══════════════════════════════════════════════════════════════════════════
const char* WIFI_SSID = "Galaxy A72AA03";
const char* WIFI_PASS = "gddu5648";

const char* SHEETS_URL =
  "https://script.google.com/macros/s/"
  "AKfycby5PRPnGOw_gBy5StMGmMqwtaCMhk8fS5BBxXRGNQNFwnslAkitmCWIJr4U2FGz6mcH/exec";

// ═══════════════════════════════════════════════════════════════════════════
//  HARDWARE PIN ASSIGNMENTS
// ═══════════════════════════════════════════════════════════════════════════
#define DS18B20_PIN        4
#define DHT_PIN            5
#define DHT_TYPE           DHT22
#define RESP_ADC_PIN       35   // GPIO35 = ADC1_CH7
#define PELTIER_RELAY_PIN  25
#define FAN_RELAY_PIN      26
#define STATUS_LED_PIN     2

#define RELAY_ON   LOW    // active-LOW relay module (most common)
#define RELAY_OFF  HIGH

// ═══════════════════════════════════════════════════════════════════════════
//  SYSTEM TIMING
// ═══════════════════════════════════════════════════════════════════════════
#define PELTIER_HOLD_MS      60000UL  // Peltier ON for 60 s after readings normalise
#define FAN_HOLD_MS         120000UL  // fans ON for 120 s after Peltier turns off
#define SENSOR_SAMPLE_MS      2000UL  // body / environmental read cycle
#define BLYNK_DASHBOARD_MS   30000L   // push to Blynk every 30 s
#define SHEETS_LOG_MS        60000L   // log to Google Sheets every 60 s

// ═══════════════════════════════════════════════════════════════════════════
//  RESPIRATORY RATE — BANDPASS + MEDIAN BPM (200 Hz non-blocking)
// ═══════════════════════════════════════════════════════════════════════════
//  Serial Plotter calibration:
//    1. Uncomment #define DEBUG_RESP_PLOTTER below
//    2. Upload → open Serial Plotter (Ctrl+Shift+L)
//    3. Three traces: bandpass_signal | threshold | rearm_level
//       - Breath peaks not crossing threshold? Lower RESP_THRESHOLD_RATIO (e.g. 0.40)
//       - Noise crossing threshold?            Raise RESP_THRESHOLD_RATIO (e.g. 0.60)
//    4. Recomment when done.
// #define DEBUG_RESP_PLOTTER

// Bandpass filter coefficients (Ts = 5 ms = 200 Hz)
//   f = α / (2π × Ts)
//   α_low  = 0.30 → high cut-off = 9.55 Hz = 573 bpm  (kills electrical noise)
//   α_high = 0.003 → low cut-off  = 0.095 Hz = 5.7 bpm (removes baseline drift)
#define RESP_ALPHA_LOW          0.30f
#define RESP_ALPHA_HIGH         0.003f

// Adaptive peak envelope
//   τ_attack = 5ms / 0.10 =   50 ms  (rises fast when signal strengthens)
//   τ_decay  = 5ms / 0.002 = 2500 ms (falls slowly when activity drops)
#define RESP_ENVELOPE_ATTACK    0.10f
#define RESP_ENVELOPE_DECAY     0.002f
#define RESP_THRESHOLD_RATIO    0.50f   // detect at 50% of peak envelope
#define RESP_REARM_RATIO        0.30f   // re-arm at 30% of threshold
#define RESP_ENVELOPE_MIN_ADC   6.0f    // absolute ADC noise floor

// Timing
#define RESP_SAMPLE_MS          5UL     // 200 Hz poll (non-blocking millis gate)
#define RESP_DEBOUNCE_MS        130UL   // max ~461 bpm — well above any dog rate
#define RESP_APNEA_MS           12000UL // reset BPM after 12 s silence
#define RESP_GAP_MIN_MS         150UL   // = 400 bpm max
#define RESP_GAP_MAX_MS         12000UL // = 5 bpm min

// Median ring buffer
#define RESP_HISTORY            5       // median of last 5 intervals

// ═══════════════════════════════════════════════════════════════════════════
//  SENSOR OBJECTS
// ═══════════════════════════════════════════════════════════════════════════
OneWire           oneWire(DS18B20_PIN);
DallasTemperature bodyTempSensor(&oneWire);
DHT               dht(DHT_PIN, DHT_TYPE);

// ═══════════════════════════════════════════════════════════════════════════
//  RESPIRATORY ADC STATE  (all private, _r prefix)
// ═══════════════════════════════════════════════════════════════════════════
static unsigned long _rLastSampleMs  = 0;

// Bandpass filter state
static float         _rLP            = 0.0f;   // low-pass accumulator
static float         _rHP            = 0.0f;   // high-pass (baseline) accumulator
static float         _rEnvelope      = 0.0f;   // peak envelope accumulator

// Peak detection state
static bool          _rPeakActive    = false;  // true while above threshold (locked out)
static bool          _rFirstBreath   = true;   // skip gap on very first detection
static unsigned long _rLastBreathMs  = 0;

// Median ring buffer
static unsigned long _rIntvl[RESP_HISTORY];
static uint8_t       _rIdx           = 0;
static uint8_t       _rCount         = 0;

// Output BPM (display-smoothed)
static float         _rBPM           = 15.0f;  // default mid-normal until first reading

// ═══════════════════════════════════════════════════════════════════════════
//  SHARED SENSOR CACHE
//  Updated by the 2-second cycle; read by Blynk and Sheets timers.
// ═══════════════════════════════════════════════════════════════════════════
float g_bodyTemp          = 38.7f;
float g_environmentalTemp = 25.0f;
float g_respRate          = 15.0f;
int   g_prediction        = 0;

// ═══════════════════════════════════════════════════════════════════════════
//  COOLING STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════════
enum CoolingState : uint8_t {
  STATE_NORMAL,
  STATE_HEAT_STRESSED,
  STATE_COOLDOWN_PELTIER,
  STATE_COOLDOWN_FAN
};
const char* stateName(CoolingState s);  // forward declaration

CoolingState  coolingState   = STATE_NORMAL;
unsigned long cooldownTimer  = 0;
unsigned long lastSensorRead = 0;

BlynkTimer blynkTimer;

// ═══════════════════════════════════════════════════════════════════════════
//  RELAY / LED HELPERS
// ═══════════════════════════════════════════════════════════════════════════
inline void setPeltier(bool on) { digitalWrite(PELTIER_RELAY_PIN, on ? RELAY_ON : RELAY_OFF); }
inline void setFan    (bool on) { digitalWrite(FAN_RELAY_PIN,     on ? RELAY_ON : RELAY_OFF); }

void activateCooling()   { setPeltier(true);  setFan(true);  digitalWrite(STATUS_LED_PIN, HIGH); }
void deactivateCooling() { setPeltier(false); setFan(false); digitalWrite(STATUS_LED_PIN, LOW);  }

const char* stateName(CoolingState s) {
  switch (s) {
    case STATE_NORMAL:           return "NORMAL";
    case STATE_HEAT_STRESSED:    return "HEAT_STRESSED";
    case STATE_COOLDOWN_PELTIER: return "COOLDOWN_PELTIER";
    case STATE_COOLDOWN_FAN:     return "COOLDOWN_FAN";
    default:                     return "UNKNOWN";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  RESPIRATORY RATE — MEDIAN HELPER
// ═══════════════════════════════════════════════════════════════════════════

// Returns BPM_median = 60000 / median(last N breath intervals)
static float computeMedianBPM() {
  unsigned long buf[RESP_HISTORY];
  for (uint8_t i = 0; i < _rCount; i++) buf[i] = _rIntvl[i];
  // Insertion sort — max 5 elements, negligible CPU cost
  for (int i = 1; i < (int)_rCount; i++) {
    unsigned long key = buf[i];
    int j = i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  unsigned long med = (_rCount % 2 == 0)
                      ? (buf[_rCount / 2 - 1] + buf[_rCount / 2]) / 2
                      :  buf[_rCount / 2];
  return 60000.0f / (float)med;
}

// ═══════════════════════════════════════════════════════════════════════════
//  RESPIRATORY RATE — MAIN SAMPLING FUNCTION
//  Must be called on EVERY loop() iteration.
//  Self-limits to 200 Hz via a 5 ms millis() gate (no delay() used).
// ═══════════════════════════════════════════════════════════════════════════
void sampleRespRate() {
  unsigned long now = millis();
  if (now - _rLastSampleMs < RESP_SAMPLE_MS) return;
  _rLastSampleMs = now;

  float raw = (float)analogRead(RESP_ADC_PIN);

  // ── FORMULA 1: Bandpass filter ──────────────────────────────────────────
  // LP[n] = α_low × x[n] + (1−α_low) × LP[n−1]
  _rLP = RESP_ALPHA_LOW  * raw + (1.0f - RESP_ALPHA_LOW)  * _rLP;
  // HP[n] = α_high × x[n] + (1−α_high) × HP[n−1]
  _rHP = RESP_ALPHA_HIGH * raw + (1.0f - RESP_ALPHA_HIGH) * _rHP;
  // BP[n] = LP[n] − HP[n]
  float bp = _rLP - _rHP;

  // ── FORMULA 2: Adaptive peak envelope ───────────────────────────────────
  float absBP = fabsf(bp);
  if (absBP > _rEnvelope) {
    // Fast attack: E[n] = α_atk × |BP| + (1−α_atk) × E[n−1]
    _rEnvelope = RESP_ENVELOPE_ATTACK * absBP + (1.0f - RESP_ENVELOPE_ATTACK) * _rEnvelope;
  } else {
    // Slow decay: E[n] = α_dec × |BP| + (1−α_dec) × E[n−1]
    _rEnvelope = RESP_ENVELOPE_DECAY  * absBP + (1.0f - RESP_ENVELOPE_DECAY)  * _rEnvelope;
  }

  float threshold  = fmaxf(RESP_ENVELOPE_MIN_ADC, _rEnvelope * RESP_THRESHOLD_RATIO);
  float rearmLevel = threshold * RESP_REARM_RATIO;

  // ── Apnea / silence timeout ─────────────────────────────────────────────
  if (_rLastBreathMs > 0 && (now - _rLastBreathMs) > RESP_APNEA_MS) {
    _rBPM        = 0.0f;
    _rFirstBreath = true;
    _rCount       = 0;
    _rIdx         = 0;
  }

  // ── Rising-edge peak detection with two-level hysteresis ────────────────
  if (!_rPeakActive && bp > threshold) {
    if (_rLastBreathMs == 0 || (now - _rLastBreathMs) > RESP_DEBOUNCE_MS) {
      if (!_rFirstBreath && _rLastBreathMs != 0) {
        unsigned long gap = now - _rLastBreathMs;
        // FORMULA 3 — only store valid breath intervals
        if (gap >= RESP_GAP_MIN_MS && gap <= RESP_GAP_MAX_MS) {
          _rIntvl[_rIdx % RESP_HISTORY] = gap;
          _rIdx++;
          if (_rCount < RESP_HISTORY) _rCount++;

          if (_rCount >= 2) {
            // BPM_median = 60000 / median(gap_1 … gap_N)
            float medBPM = computeMedianBPM();
            // BPM_out[n] = 0.15 × BPM_median + 0.85 × BPM_out[n−1]
            _rBPM = (_rBPM == 0.0f)
                    ? medBPM
                    : 0.15f * medBPM + 0.85f * _rBPM;
          }
        }
      }
      _rFirstBreath = false;
      _rLastBreathMs = now;
      _rPeakActive   = true;
    }
  }

  // Re-arm only when signal drops to 30% of threshold (prevents double-trigger)
  if (_rPeakActive && bp < rearmLevel) {
    _rPeakActive = false;
  }

#ifdef DEBUG_RESP_PLOTTER
  Serial.print(bp);         Serial.print(",");
  Serial.print(threshold);  Serial.print(",");
  Serial.println(rearmLevel);
#endif
}

float getLatestRespRate() { return _rBPM; }

// ═══════════════════════════════════════════════════════════════════════════
//  TEMPERATURE SENSOR READS
// ═══════════════════════════════════════════════════════════════════════════

float readBodyTemp() {
  bodyTempSensor.requestTemperatures();
  float t = bodyTempSensor.getTempCByIndex(0);
  // DEVICE_DISCONNECTED_C = -127 (CRC fail / no device)
  // 85.0 = DS18B20 power-on reset value — reject both
  if (t == DEVICE_DISCONNECTED_C || t == 85.0f) return -1.0f;
  return t;
}

float readEnvironmentalTemp() {
  float t = dht.readTemperature();
  return isnan(t) ? -1.0f : t;
}

// ═══════════════════════════════════════════════════════════════════════════
//  NTP TIMESTAMP HELPER
// ═══════════════════════════════════════════════════════════════════════════

String getTimestamp() {
  time_t now = time(nullptr);
  if (now < 1700000000UL) return "TIME_NOT_SYNCED";
  struct tm* t = localtime(&now);
  char buf[25];
  strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
  return String(buf);
}

// ═══════════════════════════════════════════════════════════════════════════
//  BLYNK DASHBOARD UPDATE  (called every 30 s — non-blocking)
// ═══════════════════════════════════════════════════════════════════════════

void sendBlynkData() {
  if (!Blynk.connected()) return;
  Blynk.virtualWrite(V0, g_bodyTemp);
  Blynk.virtualWrite(V1, g_respRate);
  Blynk.virtualWrite(V2, g_environmentalTemp);
  Blynk.virtualWrite(V4, g_prediction);
  Blynk.virtualWrite(V5, (int)coolingState);
}

// ═══════════════════════════════════════════════════════════════════════════
//  GOOGLE SHEETS DATA LOG  (called every 60 s)
// ═══════════════════════════════════════════════════════════════════════════

void sendSheetsData() {
  if (WiFi.status() != WL_CONNECTED) return;

  String json = "{\"ts\":\""   + getTimestamp()                + "\","
              + "\"bt\":"    + String(g_bodyTemp, 2)          + ","
              + "\"rr\":"    + String(g_respRate, 1)          + ","
              + "\"at\":"    + String(g_environmentalTemp, 2) + ","
              + "\"pred\":"  + String(g_prediction)           + ","
              + "\"state\":" + String((int)coolingState)      + "}";

  WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, SHEETS_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(8000);
  http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);

  int code = http.POST(json);
  http.end();

  if (code > 0) {
    Serial.printf("[SHEETS] Logged at %s  (HTTP %d)\n", getTimestamp().c_str(), code);
  } else {
    Serial.printf("[SHEETS] POST failed: %s\n", HTTPClient::errorToString(code).c_str());
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  COOLING STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════════

void updateCoolingState(int isStressed, unsigned long now) {
  CoolingState prev = coolingState;

  switch (coolingState) {
    case STATE_NORMAL:
      if (isStressed) { coolingState = STATE_HEAT_STRESSED; activateCooling(); }
      break;
    case STATE_HEAT_STRESSED:
      if (!isStressed) { coolingState = STATE_COOLDOWN_PELTIER; cooldownTimer = now; }
      break;
    case STATE_COOLDOWN_PELTIER:
      if (isStressed) {
        coolingState = STATE_HEAT_STRESSED;
      } else if (now - cooldownTimer >= PELTIER_HOLD_MS) {
        coolingState = STATE_COOLDOWN_FAN; cooldownTimer = now; setPeltier(false);
      }
      break;
    case STATE_COOLDOWN_FAN:
      if (isStressed) {
        coolingState = STATE_HEAT_STRESSED; activateCooling();
      } else if (now - cooldownTimer >= FAN_HOLD_MS) {
        coolingState = STATE_NORMAL; deactivateCooling();
      }
      break;
  }

  if (coolingState != prev) {
    Serial.printf("\n[STATE] %s -> %s\n", stateName(prev), stateName(coolingState));
    switch (coolingState) {
      case STATE_HEAT_STRESSED:    Serial.println("  Peltier ON + Fans ON"); break;
      case STATE_COOLDOWN_PELTIER: Serial.printf("  Peltier+Fans ON for %lu s\n", PELTIER_HOLD_MS / 1000); break;
      case STATE_COOLDOWN_FAN:     Serial.printf("  Peltier OFF. Fans ON for %lu s\n", FAN_HOLD_MS / 1000); break;
      case STATE_NORMAL:           Serial.println("  Cooldown complete. All cooling OFF."); break;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SETUP
// ═══════════════════════════════════════════════════════════════════════════
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);  // disable brownout detector
  Serial.begin(115200);
  delay(500);

  Serial.println(F("\n=========================================="));
  Serial.println(F("  Canine Heat Stroke Prevention System  "));
  Serial.println(F("=========================================="));

  // Actuators — safe off at boot
  pinMode(PELTIER_RELAY_PIN, OUTPUT);
  pinMode(FAN_RELAY_PIN,     OUTPUT);
  pinMode(STATUS_LED_PIN,    OUTPUT);
  deactivateCooling();

  // ADC — full 0–3.3 V range for piezoelectric sensor
  analogSetWidth(12);
#ifdef ADC_ATTEN_DB_12
  analogSetPinAttenuation(RESP_ADC_PIN, ADC_ATTEN_DB_12);  // ESP32 Arduino core v3.x
#else
  analogSetPinAttenuation(RESP_ADC_PIN, ADC_11db);          // core v2.x
#endif

  // Initialise bandpass filter state from first ADC reading (avoids startup spike)
  float initADC = (float)analogRead(RESP_ADC_PIN);
  _rLP = initADC;
  _rHP = initADC;

  // Temperature sensors
  bodyTempSensor.begin();
  dht.begin();
  {
    int dsCount = bodyTempSensor.getDeviceCount();
    Serial.printf("[DS18B20] %d device(s) found on OneWire bus (GPIO%d)\n", dsCount, DS18B20_PIN);
    if (dsCount == 0) {
      Serial.println(F("[DS18B20] *** NO SENSOR DETECTED ***"));
      Serial.println(F("  Most likely cause: missing 4.7kΩ pull-up resistor."));
      Serial.println(F("  FIX: connect 4.7kΩ from GPIO4 to 3.3V."));
      Serial.println(F("  DS18B20 pinout (flat face toward you): GND | DATA | VCC"));
    }
  }

  // WiFi — non-blocking with 10 s timeout
  Serial.print(F("Connecting to WiFi"));
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("WiFi OK — IP: %s\n", WiFi.localIP().toString().c_str());

    // NTP time sync — UTC+8 Philippines
    configTime(8 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    Serial.print(F("Syncing NTP"));
    for (int i = 0; i < 20 && time(nullptr) < 1700000000UL; i++) {
      delay(500); Serial.print('.');
    }
    Serial.println();

    // Blynk — 6 s timeout, auto-retry in background
    Blynk.config(BLYNK_AUTH_TOKEN);
    if (Blynk.connect(6000)) {
      Serial.println(F("[BLYNK] Connected."));
    } else {
      Serial.println(F("[BLYNK] Timed out — auto-retry active."));
      Serial.println(F("[BLYNK] Check BLYNK_TEMPLATE_ID and AUTH_TOKEN."));
      Serial.println(F("[BLYNK] Use 'Blynk IoT' app (NOT Blynk Legacy)."));
    }

    blynkTimer.setInterval(BLYNK_DASHBOARD_MS, sendBlynkData);
    blynkTimer.setInterval(SHEETS_LOG_MS,      sendSheetsData);
    Serial.println(F("Blynk + Sheets logging active."));
  } else {
    Serial.println(F("WiFi unavailable — OFFLINE mode (cooling logic unaffected)."));
  }

  Serial.println(F("\nAll sensors ready. Monitoring started.\n"));

#ifndef DEBUG_RESP_PLOTTER
  Serial.println(F("BodyTemp(C)  RespRate(bpm)  EnvTemp(C)  State                 Result         Cooldown"));
  Serial.println(F("---------------------------------------------------------------------------------------------"));
#else
  Serial.println(F("[PLOTTER MODE] bandpass_signal,threshold,rearm_level"));
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN LOOP
// ═══════════════════════════════════════════════════════════════════════════

void loop() {
  // ── 1. Respiratory ADC — runs at 200 Hz via internal 5 ms millis gate ────
  sampleRespRate();

  // ── 2. Blynk keep-alive + non-blocking timer dispatch ────────────────────
  if (WiFi.status() == WL_CONNECTED) {
    Blynk.run();
    blynkTimer.run();
  }

  // ── 3. 2-second sensor + classification cycle ─────────────────────────────
  unsigned long now = millis();
  if (now - lastSensorRead < SENSOR_SAMPLE_MS) return;
  lastSensorRead = now;

  float bodyTemp         = readBodyTemp();
  float environmentalTemp = readEnvironmentalTemp();
  float respRate         = getLatestRespRate();

  // Always update g_respRate BEFORE early returns so Blynk V1 stays current
  g_respRate = respRate;

  Serial.printf("[DEBUG] Body: %.2f C | Resp: %.1f bpm | Env: %.2f C\n",
                bodyTemp, respRate, environmentalTemp);

  if (bodyTemp < 0.0f) {
    Serial.println(F("[WARN] DS18B20 error. Add 4.7kΩ: GPIO4 --[4.7kΩ]-- 3.3V"));
    return;
  }
  if (environmentalTemp < 0.0f) {
    Serial.println(F("[WARN] DHT22 read failed — check DATA wire to GPIO5."));
    return;
  }

  g_bodyTemp          = bodyTemp;
  g_environmentalTemp = environmentalTemp;

  int prediction = classify(bodyTemp, respRate, environmentalTemp);
  g_prediction   = prediction;

  updateCoolingState(prediction, now);

#ifndef DEBUG_RESP_PLOTTER
  unsigned long elapsed = 0, target = 0;
  if (coolingState == STATE_COOLDOWN_PELTIER || coolingState == STATE_COOLDOWN_FAN) {
    elapsed = (now - cooldownTimer) / 1000UL;
    target  = (coolingState == STATE_COOLDOWN_PELTIER) ? PELTIER_HOLD_MS / 1000 : FAN_HOLD_MS / 1000;
  }
  char cdBuf[20] = "-";
  if (target) snprintf(cdBuf, sizeof(cdBuf), "%lus/%lus", elapsed, target);

  Serial.printf("%-12.2f %-14.1f %-11.2f %-21s %-14s %s\n",
                bodyTemp, respRate, environmentalTemp,
                stateName(coolingState),
                prediction ? "HEAT_STRESSED" : "NORMAL",
                cdBuf);
#endif
}
