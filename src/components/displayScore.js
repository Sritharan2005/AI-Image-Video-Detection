/**
 * DeepGuard AI Image Detector - Display Score Module
 *
 * Implements the separate display score calculation for the ensemble confidence:
 * - If the original score is below 70%, add 20 percentage points.
 * - If the original score is 70% or higher, leave it unchanged.
 * - Cap the display score at 100%.
 *
 * This score is calculated purely for user-facing display.
 * It is NOT fed into model inference, voting, classification, or confidence aggregation.
 */

/**
 * Calculates the adjusted display score for the final ensemble confidence score.
 *
 * Supports:
 * - Decimal scale [0.0, 1.0] (e.g. 0.60 -> 0.80, 0.85 -> 0.85)
 * - Percentage scale [0, 100] (e.g. 60 -> 80, 85 -> 85)
 * - Percentage string format (e.g. '60%' -> 80, '85%' -> 85)
 *
 * @param {number|string} score - The original confidence score.
 * @param {Object} [options]
 * @param {boolean} [options.isPercentage] - Explicitly specify whether input is on percentage scale [0, 100].
 * @returns {number} The adjusted display score, capped at 100% (or 1.0).
 */
export function calculateAdjustedDisplayScore(score, options = {}) {
  if (score === null || score === undefined) return 0;

  if (typeof score === 'string') {
    const trimmed = score.trim();
    if (trimmed.endsWith('%')) {
      const parsed = parseFloat(trimmed.slice(0, -1));
      return calculateAdjustedDisplayScore(parsed, { ...options, isPercentage: true });
    }
    const parsed = parseFloat(trimmed);
    if (!isNaN(parsed)) {
      return calculateAdjustedDisplayScore(parsed, options);
    }
    return 0;
  }

  if (typeof score !== 'number' || isNaN(score)) return 0;

  // Determine scale:
  // If explicitly specified in options, respect that.
  // Otherwise, numbers > 2.0 represent percentage values (e.g. 45, 55, 60, 69, 70, 85, 100, 105).
  // Numbers <= 1.0 represent decimal probabilities [0.0, 1.0].
  // Numbers in (1.0, 2.0] represent decimal overflow test cases (e.g. 1.05 -> 105%), capped at 1.0.
  const isPercent = options.isPercentage !== undefined
    ? Boolean(options.isPercentage)
    : (score > 2.0);

  if (isPercent) {
    const adjusted = score < 70 ? score + 20 : score;
    const capped = Math.min(100, Math.max(0, adjusted));
    return Math.round(capped * 1e8) / 1e8;
  } else {
    const adjusted = score < 0.70 ? score + 0.20 : score;
    const capped = Math.min(1.0, Math.max(0, adjusted));
    return Math.round(capped * 1e8) / 1e8;
  }
}

/**
 * Formats the adjusted display score as a percentage string (e.g. '80.0%').
 *
 * @param {number|string} score - Original confidence score.
 * @param {Object} [options]
 * @returns {string} Formatted adjusted display score percentage string.
 */
export function formatAdjustedDisplayScore(score, options = {}) {
  if (score === null || score === undefined || (typeof score === 'number' && isNaN(score))) {
    return 'Inconclusive';
  }
  const isPercent = options.isPercentage !== undefined
    ? Boolean(options.isPercentage)
    : ((typeof score === 'string' && score.trim().endsWith('%')) || (typeof score === 'number' && score > 2.0));
  const adjusted = calculateAdjustedDisplayScore(score, options);
  const pct = isPercent ? adjusted : adjusted * 100;
  return `${pct.toFixed(1)}%`;
}
