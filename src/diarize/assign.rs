//! Joining ASR output to diarization output.
//!
//! Pure arithmetic over timestamps: no models, no I/O, no ONNX. This is the
//! only genuinely new logic in v2, and the only part that can be tested
//! exhaustively without downloading anything.

use super::SpeakerTurn;
use crate::types::Word;
use std::collections::HashMap;

/// Length of the intersection of two half-open ranges, in seconds.
///
/// Half-open `[start, end)` means adjacency is not overlap: a word ending
/// exactly where a turn begins belongs to neither.
pub fn overlap(a_start: f32, a_end: f32, b_start: f32, b_end: f32) -> f32 {
    (a_end.min(b_end) - a_start.max(b_start)).max(0.0)
}

/// The speaker covering the most of `word`, or `None` when nothing covers it.
///
/// Overlap is summed *per speaker*, not per turn, so a speaker split across
/// several short turns is compared fairly against one with a single long turn.
///
/// Ties break toward the speaker whose earliest overlapping turn starts
/// first. Without that rule the answer would depend on the order `turns`
/// arrives in, which is not something a caller should have to reason about.
///
/// A word that overlaps nothing gets `None` rather than a guess: this crate
/// does not fabricate values it cannot compute.
pub fn speaker_for(word: &Word, turns: &[SpeakerTurn]) -> Option<usize> {
    // speaker -> (total overlap, earliest overlapping turn start)
    let mut totals: HashMap<usize, (f32, f32)> = HashMap::new();

    for turn in turns {
        // `duration()` clamps at zero, but a backwards turn would still yield
        // a bogus intersection here, so skip it outright.
        if turn.duration() <= 0.0 {
            continue;
        }
        let shared = overlap(word.start, word.end, turn.start, turn.end);
        if shared <= 0.0 {
            continue;
        }
        let entry = totals.entry(turn.speaker).or_insert((0.0, turn.start));
        entry.0 += shared;
        entry.1 = entry.1.min(turn.start);
    }

    totals
        .into_iter()
        .max_by(|(_, (a_total, a_start)), (_, (b_total, b_start))| {
            // Higher total wins; on a tie the earlier start wins, so `b_start`
            // is compared against `a_start` to invert that half of the order.
            a_total
                .partial_cmp(b_total)
                .expect("overlap totals are finite")
                .then_with(|| {
                    b_start
                        .partial_cmp(a_start)
                        .expect("turn starts are finite")
                })
        })
        .map(|(speaker, _)| speaker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Word;

    fn word(start: f32, end: f32) -> Word {
        Word { start, end, text: "x".into(), probability: 1.0 }
    }

    fn turn(start: f32, end: f32, speaker: usize) -> SpeakerTurn {
        SpeakerTurn { start, end, speaker }
    }

    #[test]
    fn disjoint_ranges_do_not_overlap() {
        assert_eq!(overlap(0.0, 1.0, 2.0, 3.0), 0.0);
    }

    #[test]
    fn touching_ranges_do_not_overlap() {
        // Ranges are half-open [start, end), so an end meeting a start is
        // adjacency, not overlap.
        assert_eq!(overlap(0.0, 1.0, 1.0, 2.0), 0.0);
    }

    #[test]
    fn partial_overlap_is_the_intersection() {
        assert_eq!(overlap(0.0, 2.0, 1.0, 3.0), 1.0);
    }

    #[test]
    fn containment_is_the_inner_range() {
        assert_eq!(overlap(0.0, 5.0, 1.0, 2.0), 1.0);
    }

    #[test]
    fn a_word_takes_the_speaker_covering_most_of_it() {
        let turns = vec![turn(0.0, 1.2, 7), turn(1.2, 3.0, 9)];
        // 0.2 s with speaker 7, 0.8 s with speaker 9.
        assert_eq!(speaker_for(&word(1.0, 2.0), &turns), Some(9));
    }

    #[test]
    fn a_word_with_no_overlapping_turn_gets_no_speaker() {
        let turns = vec![turn(10.0, 12.0, 1)];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), None);
    }

    #[test]
    fn no_turns_at_all_means_no_speaker() {
        assert_eq!(speaker_for(&word(0.0, 1.0), &[]), None);
    }

    #[test]
    fn an_exact_tie_breaks_toward_the_earlier_turn() {
        // Both cover exactly 0.5 s of the word. The result must not depend on
        // the order `turns` happens to arrive in, so it is pinned to the turn
        // that starts earlier.
        let turns = vec![turn(1.5, 2.0, 4), turn(1.0, 1.5, 2)];
        assert_eq!(speaker_for(&word(1.0, 2.0), &turns), Some(2));
    }

    #[test]
    fn overlapping_turns_for_one_speaker_accumulate() {
        // Two turns for speaker 3 covering 0.3 + 0.3, versus one turn of 0.5
        // for speaker 5. Speaker 3 wins on total, not on any single turn.
        let turns = vec![
            turn(0.0, 0.3, 3),
            turn(0.3, 0.6, 3),
            turn(0.6, 1.1, 5),
        ];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), Some(3));
    }

    #[test]
    fn a_zero_length_word_gets_no_speaker() {
        // Whisper can emit a word whose start equals its end. It overlaps
        // nothing by definition, so it is unassignable rather than assigned
        // to whichever turn happens to contain the instant.
        let turns = vec![turn(0.0, 5.0, 1)];
        assert_eq!(speaker_for(&word(2.0, 2.0), &turns), None);
    }

    #[test]
    fn a_backwards_turn_never_wins() {
        let turns = vec![turn(5.0, 1.0, 8), turn(0.0, 1.0, 2)];
        assert_eq!(speaker_for(&word(0.0, 1.0), &turns), Some(2));
    }
}
