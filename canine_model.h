// canine_model.h  —  2-of-3 Master Threshold Classifier
// Generated from: Master Threshold Classification Table (Thesis Table 2)
//
// Rule: activate cooling when >= 2 of 3 sensors are in the heat-stressed zone.
//
// Thresholds (source: thesis table, heart rate excluded):
//   Body temp        > 39.5 °C         → stressed
//   Respiratory rate >= 29 cycles/min  → stressed  (no upper cap)
//   Ambient temp     > 30.0 °C         → stressed
//
// Features must be supplied in this order:
//   body_temp        (°C)
//   respiratory_rate (cycles/min)
//   ambient_temp     (°C)

#pragma once
#include <Arduino.h>

// ── Threshold constants ───────────────────────────────────────────────────
static const float BODY_TEMP_HEAT_CUTOFF    = 39.5f;  // > this → body stressed
static const float RESP_RATE_HEAT_LOWER     = 29.0f;  // >= this → resp stressed
static const float AMBIENT_TEMP_HEAT_CUTOFF = 30.0f;  // > this → ambient stressed

// ── classify() ───────────────────────────────────────────────────────────
// Returns 1 = HEAT_STRESSED when >= 2 of 3 sensors are in stressed zone.
// Returns 0 = NORMAL otherwise.
// Inputs are raw sensor values — no scaling required.
inline int classify(float body_temp, float resp_rate, float ambient_temp) {
  int stressed = 0;
  if (body_temp    >  BODY_TEMP_HEAT_CUTOFF)    stressed++;
  if (resp_rate    >= RESP_RATE_HEAT_LOWER)      stressed++;
  if (ambient_temp >  AMBIENT_TEMP_HEAT_CUTOFF)  stressed++;
  return (stressed >= 2) ? 1 : 0;
}
