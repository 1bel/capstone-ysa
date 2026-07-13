// ESP32 Canine Respiratory Rate Monitor
// Detects full range: rest (10-29 bpm), walking/running (30-199 bpm), panting (200-400 bpm)
//
// ══════════════════════════════════════════════════════════════════════════════
// FORMULAS
// ══════════════════════════════════════════════════════════════════════════════
//
// 1. BANDPASS FILTER (two cascaded 1st-order IIR @ 200 Hz / Ts = 5 ms)
//
//    LP[n]  = α_low  × x[n] + (1 − α_low)  × LP[n−1]   ← low-pass  (noise killer)
//    HP[n]  = α_high × x[n] + (1 − α_high) × HP[n−1]   ← slow baseline tracker
//    BP[n]  = LP[n] − HP[n]                              ← bandpass = their difference
//
//    Cutoff frequencies:  f = α / (2π × Ts)
//      α_low  = 0.30  →  f_high_cut = 0.30 / (2π × 0.005) =  9.55 Hz = 573 bpm
//      α_high = 0.003 →  f_low_cut  = 0.003/ (2π × 0.005) =  0.095 Hz = 5.7 bpm
//    → Passes 5.7 bpm to 573 bpm cleanly; covers rest (10 bpm) to panting (400 bpm)
//
// 2. PEAK ENVELOPE DETECTOR (adaptive threshold, no manual tuning needed)
//
//    if |BP[n]| > E[n−1]:  E[n] = α_atk × |BP[n]| + (1−α_atk) × E[n−1]  (fast attack)
//    else:                  E[n] = α_dec × |BP[n]| + (1−α_dec) × E[n−1]  (slow decay)
//
//    τ_attack = Ts / α_atk = 5 ms / 0.10  =  50 ms   (jumps to new amplitude in ~50 ms)
//    τ_decay  = Ts / α_dec = 5 ms / 0.002 = 2500 ms  (drops to new amplitude in ~2.5 s)
//
//    Threshold  T[n] = max(T_min,  E[n] × 0.50)  ← fires when signal passes 50% of peak
//    Re-arm     R[n] = T[n] × 0.30               ← must drop to 30% before next peak counts
//
// 3. MEDIAN BPM (ring buffer of 5 intervals — immune to single outlier detections)
//
//    gap_k = t_k − t_(k−1)                 [ms between consecutive breath peaks]
//    BPM_inst_k = 60 000 / gap_k
//    BPM_median = 60 000 / median(gap_1 … gap_5)
//    BPM_display[n] = 0.15 × BPM_median + 0.85 × BPM_display[n−1]  (light smoothing)
//
// ══════════════════════════════════════════════════════════════════════════════
// HEAT STRESS THRESHOLDS (thesis)
// ══════════════════════════════════════════════════════════════════════════════
//   NORMAL    (rest):           10 – 29  bpm
//   HEAT STRESSED (rest):       30 – 64  bpm   ← stressed but not panting
//   NORMAL    (walking/running): 65 – 199 bpm  ← elevated by exercise, not heat
//   HEAT STRESSED (walking/running): 200 – 400 bpm  ← severe panting = danger
//
// ══════════════════════════════════════════════════════════════════════════════
// CALIBRATION (Serial Plotter)
// ══════════════════════════════════════════════════════════════════════════════
//   Uncomment DEBUG_PLOTTER → upload → open Serial Plotter (Ctrl+Shift+L)
//   Three traces: bandpass_signal | threshold | rearm_level
//   - Breath peaks should clearly cross the threshold line.
//   - Noise crossing threshold? Raise THRESHOLD_RATIO (e.g. 0.60) or ENVELOPE_MIN_ADC.
//   - Real breaths missed?     Lower THRESHOLD_RATIO (e.g. 0.40) or ENVELOPE_MIN_ADC.
//   Recomment when done.
// #define DEBUG_PLOTTER

#include <Arduino.h>
#include <math.h>

// ── Pin ───────────────────────────────────────────────────────────────────────
const int PIEZO_PIN = 35;

// ── Bandpass filter ───────────────────────────────────────────────────────────
// 200 Hz sample rate (delay 5 ms) required to detect 200-400 bpm panting
// α_low=0.30 → passes up to 573 bpm  |  α_high=0.003 → removes drift below 5.7 bpm
const float ALPHA_LOW  = 0.30f;
const float ALPHA_HIGH = 0.003f;
float lpFiltered = 0.0f;
float hpFiltered = 0.0f;

// ── Adaptive peak envelope ────────────────────────────────────────────────────
// Fast attack (τ≈50 ms): threshold rises immediately when signal gets stronger
// Moderate decay (τ≈2.5 s): threshold falls within ~2.5 s after activity drops
const float ENVELOPE_ATTACK   = 0.10f;
const float ENVELOPE_DECAY    = 0.002f;
const float THRESHOLD_RATIO   = 0.50f;   // detect at 50% of envelope
const float REARM_RATIO       = 0.30f;   // re-arm at 30% of threshold
const float ENVELOPE_MIN_ADC  = 6.0f;    // absolute noise floor (ADC counts)
float signalEnvelope = 0.0f;

// ── Breath detection ──────────────────────────────────────────────────────────
bool          peakDetected = false;
unsigned long lastBreathMs = 0;
bool          firstBreath  = true;

// 130 ms debounce → max detectable ≈ 461 bpm (covers 400 bpm panting)
const unsigned long DEBOUNCE_MS = 130UL;
// 12 s apnea timeout: at 10 bpm (slowest rest) a breath comes every 6 s;
// 12 s = 2× that interval, so timeout only fires when truly silent
const unsigned long APNEA_MS = 12000UL;

// ── Median BPM ring buffer ────────────────────────────────────────────────────
#define BPM_HISTORY 5
unsigned long breathIntervals[BPM_HISTORY];
uint8_t bpmIdx   = 0;
uint8_t bpmCount = 0;
float   smoothedBPM = 0.0f;

// ── Reporting ─────────────────────────────────────────────────────────────────
unsigned long lastReportMs = 0;
const unsigned long REPORT_MS = 5000UL;

// ── Compute median BPM from ring buffer ──────────────────────────────────────
float computeMedianBPM() {
  unsigned long buf[BPM_HISTORY];
  for (uint8_t i = 0; i < bpmCount; i++) buf[i] = breathIntervals[i];
  // insertion sort (max 5 elements — negligible CPU cost)
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

  // ── Bandpass filter ───────────────────────────────────────────────────────
  // LP[n] = α_low × x[n] + (1−α_low) × LP[n−1]
  lpFiltered = ALPHA_LOW  * raw + (1.0f - ALPHA_LOW)  * lpFiltered;
  // HP[n] = α_high × x[n] + (1−α_high) × HP[n−1]
  hpFiltered = ALPHA_HIGH * raw + (1.0f - ALPHA_HIGH) * hpFiltered;
  // BP[n] = LP[n] − HP[n]
  float bp = lpFiltered - hpFiltered;

  // ── Adaptive peak envelope → threshold ───────────────────────────────────
  float absBP = fabsf(bp);
  if (absBP > signalEnvelope) {
    // Fast attack: E[n] = α_atk × |BP| + (1−α_atk) × E[n−1]
    signalEnvelope = ENVELOPE_ATTACK * absBP + (1.0f - ENVELOPE_ATTACK) * signalEnvelope;
  } else {
    // Slow decay:  E[n] = α_dec × |BP| + (1−α_dec) × E[n−1]
    signalEnvelope = ENVELOPE_DECAY  * absBP + (1.0f - ENVELOPE_DECAY)  * signalEnvelope;
  }

  float threshold  = fmaxf(ENVELOPE_MIN_ADC, signalEnvelope * THRESHOLD_RATIO);
  float rearmLevel = threshold * REARM_RATIO;

  // ── Apnea / silence timeout ──────────────────────────────────────────────
  if (lastBreathMs > 0 && (now - lastBreathMs) > APNEA_MS) {
    smoothedBPM = 0.0f;
    firstBreath = true;
    bpmCount    = 0;
    bpmIdx      = 0;
  }

  // ── Rising-edge peak detection with two-level hysteresis ─────────────────
  if (!peakDetected && bp > threshold) {
    if (lastBreathMs == 0 || (now - lastBreathMs) > DEBOUNCE_MS) {
      if (!firstBreath && lastBreathMs != 0) {
        unsigned long gap = now - lastBreathMs;
        // Valid range: 60000/400bpm=150 ms  to  60000/5bpm=12000 ms
        if (gap >= 150UL && gap <= 12000UL) {
          // Store interval in ring buffer
          breathIntervals[bpmIdx % BPM_HISTORY] = gap;
          bpmIdx++;
          if (bpmCount < BPM_HISTORY) bpmCount++;

          if (bpmCount >= 2) {
            // BPM_median = 60000 / median(last 5 gaps)
            float medBPM = computeMedianBPM();
            // BPM_display[n] = 0.15 × BPM_median + 0.85 × BPM_display[n-1]
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

  // Re-arm only after signal drops to 30% of threshold (prevents double-trigger)
  if (peakDetected && bp < rearmLevel) {
    peakDetected = false;
  }

#ifdef DEBUG_PLOTTER
  // Serial Plotter: three traces for calibration
  Serial.print(bp);          Serial.print(",");
  Serial.print(threshold);   Serial.print(",");
  Serial.println(rearmLevel);
#endif

  // ── Timed BPM report every 5 s ───────────────────────────────────────────
#ifndef DEBUG_PLOTTER
  if (now - lastReportMs >= REPORT_MS) {
    lastReportMs = now;

    int bpm = (int)roundf(smoothedBPM);
    if (bpm > 400) bpm = 400;  // physical cap

    int displayBPM = bpm;
    if (displayBPM > 64) {
      displayBPM = 64;
    }

    Serial.print(F("BPM: "));
    Serial.print(displayBPM);
    Serial.print(F(" | Signal: "));
    Serial.print(signalEnvelope, 1);
    Serial.print(F(" | Threshold: "));
    Serial.print(threshold, 1);
    Serial.print(F(" | Status: "));

    // ── Heat stress classification ──────────────────────────────────────────
    //   NORMAL    rest:           10 – 29  bpm
    //   HEAT STRESSED (rest):     30 – 64  bpm   (stressed but not panting)
    //   NORMAL    active:         65 – 199 bpm   (elevated by exercise, not heat)
    //   HEAT STRESSED (active):  200 – 400 bpm   (severe panting = heat danger)
    if (bpm == 0) {
      Serial.println(F("INITIALIZING / NO SIGNAL"));
    } else if (bpm <= 29) {
      Serial.println(F("NORMAL (Rest)"));
    } else if (bpm <= 64) {
      Serial.println(F("HEAT STRESSED (Rest — 30-64 bpm)"));
    } else if (bpm <= 199) {
      Serial.println(F("NORMAL (Active / Walking / Running)"));
    } else {
      Serial.println(F("HEAT STRESSED (Active — panting 200-400 bpm)"));
    }
  }
#endif

  delay(5);   // 200 Hz sampling — required to detect 200-400 bpm panting
              // (50 Hz / delay 20 was too slow: max filter passband was only 72 bpm)
}
