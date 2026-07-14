/*
 * esp32_cooling_system.ino
 * ─────────────────────────────────────────────────────────────────────────
 * Canine Heat Stroke Prevention System — Blynk IoT Edition (Dual-Core Optimized)
 * Thesis: "IoT Monitoring of Vital Signs and Active Cooling for
 * Canine Heat Stroke Prevention Using Binary Classification"
 */

// ═══════════════════════════════════════════════════════════════════════════
//  BLYNK CREDENTIALS
// ═══════════════════════════════════════════════════════════════════════════
#define BLYNK_TEMPLATE_ID    "TMPL6ON4E1cJ3"           
#define BLYNK_TEMPLATE_NAME  "Canine Cooling System"
#define BLYNK_AUTH_TOKEN     "agAVZhjCKKq8kUqQr-3hBbVFFZn55hzh"  
#define BLYNK_PRINT          Serial

// ═══════════════════════════════════════════════════════════════════════════
//  INCLUDES
// ═══════════════════════════════════════════════════════════════════════════
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>   
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <time.h>
#include <math.h>

#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include "canine_model.h"  

// ═══════════════════════════════════════════════════════════════════════════
//  NETWORK CREDENTIALS
// ═══════════════════════════════════════════════════════════════════════════
const char* WIFI_SSID = "HUAWEI-6HAd";
const char* WIFI_PASS = "YUm5TMfW";
const char* SHEETS_URL = "https://script.google.com/macros/s/AKfycby5PRPnGOw_gBy5StMGmMqwtaCMhk8fS5BBxXRGNQNFwnslAkitmCWIJr4U2FGz6mcH/exec";

// ═══════════════════════════════════════════════════════════════════════════
//  HARDWARE PIN ASSIGNMENTS
// ═══════════════════════════════════════════════════════════════════════════
#define DS18B20_PIN        4
#define DHT_PIN            5
#define DHT_TYPE           DHT22
#define RESP_ADC_PIN       35   
#define PELTIER_RELAY_PIN  25
#define FAN_RELAY_PIN      26
#define STATUS_LED_PIN     2    

#define RELAY_ON   HIGH    
#define RELAY_OFF  LOW   

// ═══════════════════════════════════════════════════════════════════════════
//  TIMING CONSTRAINTS
// ═══════════════════════════════════════════════════════════════════════════
#define PELTIER_HOLD_MS     60000UL  
#define FAN_HOLD_MS         120000UL  
#define SENSOR_SAMPLE_MS      2000UL  

// Optimized Task Intervals
#define BLYNK_DASHBOARD_MS    2000UL   // Sped up to 2 seconds for ultra-responsive Blynk Gauges
#define SHEETS_LOG_MS        60000UL   // Logs data to sheets every 60 seconds

// ═══════════════════════════════════════════════════════════════════════════
//  RESPIRATORY CONFIGURATION (200 Hz Advanced Detection Engine)
// ═══════════════════════════════════════════════════════════════════════════
const unsigned long RESP_SAMPLE_MS = 5UL;
const float ALPHA_LOW  = 0.30f;
const float ALPHA_HIGH = 0.003f;
static float lpFiltered = 0.0f;
static float hpFiltered = 0.0f;

const float ENVELOPE_ATTACK   = 0.10f;
const float ENVELOPE_DECAY    = 0.002f;
const float THRESHOLD_RATIO   = 0.50f;
const float REARM_RATIO       = 0.30f;
const float ENVELOPE_MIN_ADC  = 6.0f;
static float signalEnvelope   = 0.0f;

bool peakDetected = false;
unsigned long lastBreathMs = 0;
bool firstBreath = true;
const unsigned long DEBOUNCE_MS = 130UL;
const unsigned long APNEA_MS = 12000UL;

#define BPM_HISTORY 5
static unsigned long breathIntervals[BPM_HISTORY];
static uint8_t bpmIdx   = 0;
static uint8_t bpmCount = 0;
static float smoothedBPM = 0.0f;

// ═══════════════════════════════════════════════════════════════════════════
//  EXPLICIT STATE MACHINE SCHEME (BLYNK GAUGE COMPATIBLE)
// ═══════════════════════════════════════════════════════════════════════════
enum CoolingState : uint8_t {
  STATE_HEAT_STRESSED    = 0,  // All Cooling Hardware Active (Peltier & Fans ON)
  STATE_COOLDOWN_PELTIER = 1,  // Safe Run Hold Phase (Peltier & Fans remain ON)
  STATE_COOLDOWN_FAN     = 2,  // Heat Extraction Phase (Peltier OFF, Fans stay ON)
  STATE_NORMAL           = 3   // Safe Passive Window (All Cooling Hardware OFF)
};
const char* stateName(CoolingState s);   

CoolingState  coolingState   = STATE_NORMAL;
unsigned long cooldownTimer  = 0;
unsigned long lastSensorRead = 0;

// ═══════════════════════════════════════════════════════════════════════════
//  SENSOR OBJECTS & HARDWARE SHORTCUTS
// ═══════════════════════════════════════════════════════════════════════════
OneWire           oneWire(DS18B20_PIN);
DallasTemperature bodyTempSensor(&oneWire);
DHT               dht(DHT_PIN, DHT_TYPE);

inline void setPeltier(bool on) { digitalWrite(PELTIER_RELAY_PIN, on ? RELAY_ON : RELAY_OFF); }
inline void setFan    (bool on) { digitalWrite(FAN_RELAY_PIN,     on ? RELAY_ON : RELAY_OFF); }

void activateCooling()   { setPeltier(true);  setFan(true);  digitalWrite(STATUS_LED_PIN, HIGH); }
void deactivateCooling() { setPeltier(false); setFan(false); digitalWrite(STATUS_LED_PIN, LOW); }

const char* stateName(CoolingState s) {
  switch (s) {
    case STATE_HEAT_STRESSED:    return "0: HEAT_STRESSED_ALL_ON";
    case STATE_COOLDOWN_PELTIER: return "1: COOLDOWN_PELTIER";
    case STATE_COOLDOWN_FAN:     return "2: COOLDOWN_FAN";
    case STATE_NORMAL:           return "3: NORMAL_ALL_OFF";
    default:                     return "UNKNOWN";
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  FREE RTOS DUAL-CORE MUTEX & SYNCHRONIZATION DATA SHIELD
// ═══════════════════════════════════════════════════════════════════════════
struct TelemetryData {
  float bodyTemp;
  float respRate;
  float envTemp;
  int prediction;
  int coolingState;
};

TelemetryData syncData = {38.7f, 15.0f, 25.0f, 0, 3};
portMUX_TYPE dataMutex = portMUX_INITIALIZER_UNLOCKED;
TaskHandle_t TelemetryTaskHandle;

void TelemetryWorker(void * pvParameters);

// ═══════════════════════════════════════════════════════════════════════════
//  INTEGRATED 200Hz RESPIRATORY RATE PROCESSOR
// ═══════════════════════════════════════════════════════════════════════════
float computeMedianBPM() {
  unsigned long buf[BPM_HISTORY];
  for (uint8_t i = 0; i < bpmCount; i++) buf[i] = breathIntervals[i];
  for (int i = 1; i < (int)bpmCount; i++) {
    unsigned long key = buf[i];
    int j = i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  
  unsigned long med = (bpmCount % 2 == 0)
                      ? (buf[bpmCount / 2 - 1] + buf[bpmCount / 2]) / 2
                      : buf[bpmCount / 2];
  return 60000.0f / (float)med;
}

void sampleRespRate() {
  static unsigned long _rLastSampleMs = 0;
  unsigned long now = millis();
  if (now - _rLastSampleMs < RESP_SAMPLE_MS) return;
  _rLastSampleMs = now;

  float raw = (float)analogRead(RESP_ADC_PIN);
  lpFiltered = ALPHA_LOW  * raw + (1.0f - ALPHA_LOW)  * lpFiltered;
  hpFiltered = ALPHA_HIGH * raw + (1.0f - ALPHA_HIGH) * hpFiltered;
  float bp = lpFiltered - hpFiltered;
  float absBP = fabsf(bp);
  if (absBP > signalEnvelope) {
    signalEnvelope = ENVELOPE_ATTACK * absBP + (1.0f - ENVELOPE_ATTACK) * signalEnvelope;
  } else {
    signalEnvelope = ENVELOPE_DECAY  * absBP + (1.0f - ENVELOPE_DECAY)  * signalEnvelope;
  }

  float threshold  = fmaxf(ENVELOPE_MIN_ADC, signalEnvelope * THRESHOLD_RATIO);
  float rearmLevel = threshold * REARM_RATIO;
  if (lastBreathMs > 0 && (now - lastBreathMs) > APNEA_MS) {
    smoothedBPM = 0.0f;
    firstBreath = true;
    bpmCount    = 0;
    bpmIdx      = 0;
  }

  if (!peakDetected && bp > threshold) {
    if (lastBreathMs == 0 || (now - lastBreathMs) > DEBOUNCE_MS) {
      if (!firstBreath && lastBreathMs != 0) {
        unsigned long gap = now - lastBreathMs;
        if (gap >= 150UL && gap <= 12000UL) {
          breathIntervals[bpmIdx % BPM_HISTORY] = gap;
          bpmIdx++;
          if (bpmCount < BPM_HISTORY) bpmCount++;

          if (bpmCount >= 2) {
            float medBPM = computeMedianBPM();
            smoothedBPM = (smoothedBPM == 0.0f) ? medBPM : 0.15f * medBPM + 0.85f * smoothedBPM;
          }
        }
      }
      firstBreath  = false;
      lastBreathMs = now;
      peakDetected = true;
    }
  }

  if (peakDetected && bp < rearmLevel) {
    peakDetected = false;
  }
}

float getLatestRespRate() { 
  int bpm = (int)roundf(smoothedBPM);
  if (bpm > 400) bpm = 400; 
  return (float)bpm;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ENVIRONMENTAL SENSOR READERS
// ═══════════════════════════════════════════════════════════════════════════
float readBodyTemp() {
  bodyTempSensor.requestTemperatures();
  float t = bodyTempSensor.getTempCByIndex(0);
  if (t == DEVICE_DISCONNECTED_C || t == 85.0f) return -1.0f;
  return t;
}

float readenvironmentalTemp() {
  float t = dht.readTemperature();
  return isnan(t) ? -1.0f : t;
}

// ═══════════════════════════════════════════════════════════════════════════
//  ANTI-OSCILLATION AUTOMATED STATE MACHINE
// ═══════════════════════════════════════════════════════════════════════════
void updateCoolingState(int isStressed, unsigned long now) {
  CoolingState prev = coolingState;
  switch (coolingState) {
    case STATE_NORMAL: 
      if (isStressed) { 
        coolingState = STATE_HEAT_STRESSED;
        activateCooling(); 
      }
      break;
    case STATE_HEAT_STRESSED: 
      if (!isStressed) { 
        coolingState = STATE_COOLDOWN_PELTIER;
        cooldownTimer = now;
      }
      break;
    case STATE_COOLDOWN_PELTIER: 
      if (isStressed) {
        coolingState = STATE_HEAT_STRESSED;
        digitalWrite(STATUS_LED_PIN, HIGH);
        Serial.println(F("[ANTI-OSCILLATION] Re-triggered heat stress during Peltier Cooldown! Resetting state to 0."));
      } else if (now - cooldownTimer >= PELTIER_HOLD_MS) {
        coolingState = STATE_COOLDOWN_FAN;
        cooldownTimer = now; 
        setPeltier(false);
      }
      break;
    case STATE_COOLDOWN_FAN: 
      if (isStressed) {
        coolingState = STATE_HEAT_STRESSED;
        activateCooling();
        Serial.println(F("[ANTI-OSCILLATION] Re-triggered heat stress during Fan Cooldown! Restarting full cooling."));
      } else if (now - cooldownTimer >= FAN_HOLD_MS) {
        coolingState = STATE_NORMAL; 
        deactivateCooling();
      }
      break;
  }

  if (coolingState != prev) {
    Serial.printf("[STATE SHIFT] Changed from %s to %s\n", stateName(prev), stateName(coolingState));
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SETUP (EXECUTED BY SYSTEM ON DEFAULT CORE 1)
// ═══════════════════════════════════════════════════════════════════════════
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"

void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);
  Serial.begin(115200);
  delay(500);

  pinMode(PELTIER_RELAY_PIN, OUTPUT);
  pinMode(FAN_RELAY_PIN,     OUTPUT);
  pinMode(STATUS_LED_PIN,    OUTPUT);
  deactivateCooling(); 

  analogSetWidth(12);
#ifdef ADC_ATTEN_DB_12
  analogSetPinAttenuation(RESP_ADC_PIN, ADC_ATTEN_DB_12);
#else
  analogSetPinAttenuation(RESP_ADC_PIN, ADC_11db);
#endif

  // FIX #1: Set DallasTemperature to non-blocking mode
  bodyTempSensor.begin();
  bodyTempSensor.setWaitForConversion(false); 
  dht.begin();

  float initVal = (float)analogRead(RESP_ADC_PIN);
  lpFiltered = initVal;
  hpFiltered = initVal;
  for (int i = 0; i < BPM_HISTORY; i++) breathIntervals[i] = 2000;

  Serial.print(F("Connecting to WiFi"));
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  for (int i = 0; i < 20 && WiFi.status() != WL_CONNECTED; i++) {
    delay(500); Serial.print('.');
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    configTime(8 * 3600, 0, "pool.ntp.org", "time.nist.gov");
    Blynk.config(BLYNK_AUTH_TOKEN);
    Blynk.connect(6000);
  }

  xTaskCreatePinnedToCore(
    TelemetryWorker,         
    "TelemetryTask",         
    8192,                    
    NULL,                    
    1,                       
    &TelemetryTaskHandle,    
    0                        
  );
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN LOOP (CORE 1 CONTINUOUS HARDWARE CONTROL ENGINE)
// ═══════════════════════════════════════════════════════════════════════════
void loop() {
  // Always sample the 200Hz respiratory wave first 
  sampleRespRate();
  
  unsigned long now = millis();
  if (now - lastSensorRead < SENSOR_SAMPLE_MS) return;
  lastSensorRead = now;
  
  float bodyTemp    = readBodyTemp();
  float environmentalTemp = readenvironmentalTemp();
  float respRate    = getLatestRespRate();
  
  // FIX #2: Graceful error logging if sensors disconnect, preventing a system lockup
  if (bodyTemp < 0.0f || environmentalTemp < 0.0f) {
    Serial.printf("[WARN] Sensor Fault Detected! BodyTemp: %.1f C | EnvTemp: %.1f C\n", bodyTemp, environmentalTemp);
    return; // Returns out of this execution block safely, allowing loop() to restart cleanly
  }
  
  int prediction = classify(bodyTemp, respRate, environmentalTemp);
  updateCoolingState(prediction, now);

  portENTER_CRITICAL(&dataMutex);
  syncData.bodyTemp = bodyTemp;
  syncData.respRate = respRate;
  syncData.envTemp = environmentalTemp;
  syncData.prediction = prediction;
  syncData.coolingState = (int)coolingState;
  portEXIT_CRITICAL(&dataMutex);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CORE 0 TELEMETRY WORKER (ISOLATED CLOUD PROCESSING FACTORY)
// ═══════════════════════════════════════════════════════════════════════════
void TelemetryWorker(void * pvParameters) {
  unsigned long lastBlynkMs = 0;
  unsigned long lastSheetsMs = 0;

  vTaskDelay(pdMS_TO_TICKS(1000));
  for(;;) {
    vTaskDelay(pdMS_TO_TICKS(10));
    
    if (WiFi.status() == WL_CONNECTED && Blynk.connected()) {
      Blynk.run();
    }

    unsigned long now = millis();
    TelemetryData dataSnapshot;
    
    portENTER_CRITICAL(&dataMutex);
    dataSnapshot = syncData;
    portEXIT_CRITICAL(&dataMutex);
    
    if (now - lastBlynkMs >= BLYNK_DASHBOARD_MS) {
      lastBlynkMs = now;
      if (WiFi.status() == WL_CONNECTED && Blynk.connected()) {
        Blynk.virtualWrite(V0, dataSnapshot.bodyTemp);
        Blynk.virtualWrite(V1, dataSnapshot.respRate);
        Blynk.virtualWrite(V2, dataSnapshot.envTemp);
        Blynk.virtualWrite(V3, dataSnapshot.prediction);   
        Blynk.virtualWrite(V4, dataSnapshot.coolingState);
      }
    }

    if (now - lastSheetsMs >= SHEETS_LOG_MS) {
      lastSheetsMs = now;
      if (WiFi.status() == WL_CONNECTED) {
        time_t rawtime = time(nullptr);
        String ts = "TIME_NOT_SYNCED";
        if (rawtime >= 1700000000UL) {
          struct tm* t = localtime(&rawtime);
          char buf[25];
          strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", t);
          ts = String(buf);
        }

        String json = "{\"ts\":\""   + ts + "\","
                      + "\"bt\":"    + String(dataSnapshot.bodyTemp, 2) + ","
                      + "\"rr\":"    + String(dataSnapshot.respRate, 1) + ","
                      + "\"at\":"    + String(dataSnapshot.envTemp, 2) + ","
                      + "\"pred\":"  + String(dataSnapshot.prediction) + ","
                      + "\"state\":" + String(dataSnapshot.coolingState) + "}";
                      
        WiFiClientSecure client;
        client.setInsecure();    
        HTTPClient http;
        http.begin(client, SHEETS_URL);
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(1500); 
        http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);  

        int code = http.POST(json);
        http.end();
        if (code > 0) {
          Serial.printf("[ASYNC SHEETS] Row logged via Core 0 at %s (HTTP %d)\n", ts.c_str(), code);
        } else {
          Serial.printf("[ASYNC SHEETS] POST Failed: %s\n", HTTPClient::errorToString(code).c_str());
        }
      }
    }
  }
}