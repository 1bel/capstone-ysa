// ESP32 Canine Respiratory Rate Monitor
// Accurate across rest (10-28 bpm), walking (~20-40 bpm), running (~30-64 bpm)
#include <Arduino.h>
#include <math.h>
//
// Algorithm:
//   1. Bandpass filter  — low cutoff ~1.4 bpm (removes drift), high cutoff ~72 bpm (kills noise)
//   2. Peak envelope    — fast attack / slow decay adaptive threshold; no manual tuning needed
//   3. Median BPM       — ring buffer of 5 intervals, median is immune to single outlier detections
//   4. Light display EMA (alpha=0.15) — smooth number on Serial without lag
//
// ── Serial Plotter calibration ───────────────────────────────────────────────
// Uncomment DEBUG_PLOTTER, upload, open Serial Plotter (Ctrl+Shift+L).
// You will see three lines: bandpass_signal | threshold | re-arm_level
//   - Breath peaks should clearly cross the threshold line.
//   - If noise is crossing: raise THRESHOLD_RATIO (e.g. 0.6) or ENVELOPE_MIN_ADC.
//   - If real breaths are missed: lower THRESHOLD_RATIO (e.g. 0.4) or ENVELOPE_MIN_ADC.
// Recomment when done.
// #define DEBUG_PLOTTER

const int PIEZO_PIN = 35;

// ── Bandpass filter coefficients ─────────────────────────────────────────────
// lowPass  alpha=0.15 → -3 dB at ~71.6 bpm  — passes all breathing, kills mains noise
// highPass alpha=0.003 → -3 dB at ~1.4 bpm  — removes only slow baseline/temperature drift
// FIX: original alpha_high=0.05 cut off rest breathing at <24 bpm
const float ALPHA_LOW  = 0.15f;
const float ALPHA_HIGH = 0.003f;
float lpFiltered = 0.0f;
float hpFiltered = 0.0f;

// ── Adaptive peak envelope ────────────────────────────────────────────────────
// Tracks signal amplitude: fast attack so threshold rises quickly on motion/run start,
// moderate decay so threshold falls within ~4 s when dog stops running.
// Threshold = envelope * THRESHOLD_RATIO — always relative to current signal strength.
const float ENVELOPE_ATTACK = 0.10f;    // tau ~200 ms — reacts to new stronger signal quickly
const float ENVELOPE_DECAY  = 0.005f;   // tau ~4 s   — drops within 4 s after activity stops
const float THRESHOLD_RATIO = 0.50f;    // detect at 50% of envelope peak
const float REARM_RATIO     = 0.30f;    // re-arm when signal drops to 30% of threshold
const float ENVELOPE_MIN_ADC = 6.0f;    // absolute minimum threshold (ADC counts)
float signalEnvelope = 0.0f;

// ── Breath detection state ────────────────────────────────────────────────────
bool          peakDetected  = false;
unsigned long lastBreathMs  = 0;
bool          firstBreath   = true;

const unsigned long DEBOUNCE_MS   = 500UL;   // max ~120 bpm — well above any dog breathing
const unsigned long APNEA_MS      = 12000UL; // reset BPM if silent for 12 s (rest ~10 bpm = 6 s/breath)

// ── Ring buffer: median of last 5 intervals (FIX: replaces EMA-on-instantBPM) ─
#define BPM_HISTORY 5
unsigned long breathIntervals[BPM_HISTORY];
uint8_t bpmIdx   = 0;
uint8_t bpmCount = 0;
float   smoothedBPM = 0.0f;   // display value; light EMA on median

// ── Reporting ─────────────────────────────────────────────────────────────────
unsigned long lastReportMs = 0;
const unsigned long REPORT_MS = 5000UL;

// ── Median BPM helper ─────────────────────────────────────────────────────────
float computeMedianBPM() {
  unsigned long buf[BPM_HISTORY];
  for (uint8_t i = 0; i < bpmCount; i++) buf[i] = breathIntervals[i];
  // insertion sort (bpmCount <= 5, so trivially fast)
  for (int i = 1; i < (int)bpmCount; i++) {
    unsigned long key = buf[i];
    int j = i - 1;
    while (j >= 0 && buf[j] > key) { buf[j + 1] = buf[j]; j--; }
    buf[j + 1] = key;
  }
  unsigned long med = (bpmCount % 2 == 0)
                      ? (buf[bpmCount / 2 - 1] + buf[bpmCount / 2]) / 2
                      :  buf[bpmCount / 2];
  return 60000.0f / (float)med;
}

void setup() {
  Serial.begin(115200);
  pinMode(PIEZO_PIN, INPUT);

  float init = (float)analogRead(PIEZO_PIN);
  lpFiltered = init;
  hpFiltered = init;

  lastReportMs = millis();

#ifdef DEBUG_PLOTTER
  Serial.println(F("bandpass_signal,threshold,rearm_level"));
#endif
}

void loop() {
  unsigned long now = millis();
  float raw = (float)analogRead(PIEZO_PIN);

  // ── Stage 1: Bandpass filter ─────────────────────────────────────────────
  lpFiltered = ALPHA_LOW  * raw + (1.0f - ALPHA_LOW)  * lpFiltered;
  hpFiltered = ALPHA_HIGH * raw + (1.0f - ALPHA_HIGH) * hpFiltered;
  float bp = lpFiltered - hpFiltered;

  // ── Stage 2: Peak envelope → adaptive threshold ──────────────────────────
  float absBP = fabsf(bp);
  if (absBP > signalEnvelope) {
    signalEnvelope = ENVELOPE_ATTACK * absBP + (1.0f - ENVELOPE_ATTACK) * signalEnvelope;
  } else {
    signalEnvelope = ENVELOPE_DECAY  * absBP + (1.0f - ENVELOPE_DECAY)  * signalEnvelope;
  }

  float threshold = fmaxf(ENVELOPE_MIN_ADC, signalEnvelope * THRESHOLD_RATIO);
  float rearmLevel = threshold * REARM_RATIO;

  // ── Stage 3: Apnea / silence timeout ────────────────────────────────────
  if (lastBreathMs > 0 && (now - lastBreathMs) > APNEA_MS) {
    smoothedBPM = 0.0f;
    firstBreath = true;
    bpmCount    = 0;
    bpmIdx      = 0;
  }

  // ── Stage 4: Rising-edge peak detection with two-level hysteresis ────────
  if (!peakDetected && bp > threshold) {
    if (lastBreathMs == 0 || (now - lastBreathMs) > DEBOUNCE_MS) {
      if (!firstBreath && lastBreathMs != 0) {
        unsigned long gap = now - lastBreathMs;
        // Valid breath interval: 60000/5bpm=12000ms down to 60000/100bpm=600ms
        if (gap >= 600UL && gap <= 12000UL) {
          breathIntervals[bpmIdx % BPM_HISTORY] = gap;
          bpmIdx++;
          if (bpmCount < BPM_HISTORY) bpmCount++;

          // ── Stage 5: Median of ring buffer ── immune to single outlier ──
          if (bpmCount >= 2) {
            float medBPM = computeMedianBPM();
            // Light EMA on median only for smooth display — does NOT cause BPM spikes
            smoothedBPM = (smoothedBPM == 0.0f)
                          ? medBPM
                          : 0.15f * medBPM + 0.85f * smoothedBPM;
          }
        }
      }
      firstBreath  = false;
      lastBreathMs = now;
      peakDetected = true;
    }
  }

  // Re-arm only after signal drops well below threshold (prevents double-trigger)
  if (peakDetected && bp < rearmLevel) {
    peakDetected = false;
  }

  // ── Plotter output (uncomment DEBUG_PLOTTER at top to enable) ────────────
#ifdef DEBUG_PLOTTER
  Serial.print(bp);          Serial.print(",");
  Serial.print(threshold);   Serial.print(",");
  Serial.println(rearmLevel);
#endif

  // ── Timed BPM report every 5 s ───────────────────────────────────────────
#ifndef DEBUG_PLOTTER
  if (now - lastReportMs >= REPORT_MS) {
    lastReportMs = now;

    int bpm = (int)roundf(smoothedBPM);
    if (bpm > 64) bpm = 64;  // cap at project max

    Serial.print(F("BPM: "));
    Serial.print(bpm);
    Serial.print(F(" | Envelope: "));
    Serial.print(signalEnvelope, 1);
    Serial.print(F(" | Threshold: "));
    Serial.print(threshold, 1);
    Serial.print(F(" | Status: "));

    if (bpm == 0) {
      Serial.println(F("INITIALIZING / NO SIGNAL"));
    } else if (bpm <= 28) {
      Serial.println(F("NORMAL (Rest)"));        // thesis: resp < 29 -> normal
    } else if (bpm <= 64) {
      Serial.println(F("ELEVATED / HEAT STRESSED"));  // thesis: resp >= 29 -> stressed
    } else {
      Serial.println(F("OUT OF BOUNDS"));
    }
  }
#endif

  delay(20);  // 50 Hz sampling — fine for this standalone sketch
}
