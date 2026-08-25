//! Joining ASR output to diarization output.
//!
//! Pure arithmetic over timestamps: no models, no I/O, no ONNX. This is the
//! only genuinely new logic in v2, and the only part that can be tested
//! exhaustively without downloading anything.

use super::SpeakerTurn;
use crate::types::{Seg, Word};
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
        .max_by(|(a_id, (a_total, a_start)), (b_id, (b_total, b_start))| {
            // Higher total wins; on a tie the earlier start wins, so `b_start`
            // is compared against `a_start` to invert that half of the order.
            // The speaker id breaks a remaining exact tie: without it two
            // speakers with identical overlap AND identical earliest start
            // resolve by HashMap iteration order, which is the
            // nondeterminism this ordering exists to remove -- the same
            // audio could then produce different labels run to run.
            a_total
                .partial_cmp(b_total)
                .expect("overlap totals are finite")
                .then_with(|| {
                    b_start
                        .partial_cmp(a_start)
                        .expect("turn starts are finite")
                })
                .then_with(|| b_id.cmp(a_id))
        })
        .map(|(speaker, _)| speaker)
}

/// Rebuild readable text from words whose original spacing is gone.
///
/// `ct2rs` trims every word before returning it (`whisper.rs`, where the
/// decoded token text is `.trim()`ed), so a word never carries the leading
/// space Whisper's own segment text has. Concatenating them directly is
/// therefore not "close enough" -- it produces `Thisisspeakernumberzero`,
/// which was what a run of the documented README example actually printed.
///
/// The exact original spacing is unrecoverable here: it was discarded
/// upstream, before this crate saw the words. So this rejoins on the usual
/// convention -- one space between words, and none before a token that
/// attaches to the word it follows (closing punctuation, contractions). It is
/// a heuristic, and it is only ever applied to segments that were SPLIT;
/// unsplit segments keep the ASR's own text untouched.
fn join_words(words: &[Word]) -> String {
    let mut text = String::new();
    for word in words {
        let attaches = word
            .text
            .chars()
            .next()
            .is_none_or(|c| ",.!?;:%)]}'\u{2019}".contains(c));
        if !text.is_empty() && !attaches {
            text.push(' ');
        }
        text.push_str(&word.text);
    }
    text
}

/// Assign a speaker to every word, then split each segment into runs of
/// consecutive same-speaker words.
///
/// Returned segments carry `id: 0`; sequential numbering is applied later, by
/// `stitch::number`, because splitting changes how many segments exist and the
/// iterator that hands them out is lazy.
///
/// A split segment's `text` is rebuilt from its words by `join_words`, so it
/// may differ from the original in whitespace. A segment that is *not* split
/// keeps its original text verbatim.
pub fn assign(segs: Vec<Seg>, turns: &[SpeakerTurn]) -> Vec<Seg> {
    let mut out = Vec::with_capacity(segs.len());

    for mut seg in segs {
        let Some(mut words) = seg.words.take() else {
            // No word alignment: nothing to assign, nothing to split on.
            out.push(Seg { speaker: None, words: None, ..seg });
            continue;
        };

        if words.is_empty() {
            out.push(Seg { speaker: None, words: Some(words), ..seg });
            continue;
        }

        for word in &mut words {
            word.speaker = speaker_for(word, turns);
        }

        // One output segment per run of consecutive words sharing a speaker.
        // Runs of `None` are runs too, so unattributed speech stays visible
        // instead of being folded into a neighbour.
        let mut runs: Vec<Vec<Word>> = Vec::new();
        for word in words {
            match runs.last_mut() {
                Some(run) if run[0].speaker == word.speaker => run.push(word),
                _ => runs.push(vec![word]),
            }
        }

        if runs.len() == 1 {
            // Unsplit: keep the original text exactly as the ASR produced it.
            let speaker = runs[0][0].speaker;
            out.push(Seg { speaker, words: Some(runs.pop().unwrap()), ..seg });
            continue;
        }

        for run in runs {
            let speaker = run[0].speaker;
            let start = run.first().expect("a run is never empty").start;
            let end = run.last().expect("a run is never empty").end;
            let text = join_words(&run);
            out.push(Seg { id: 0, start, end, text, words: Some(run), speaker });
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Word;

    fn word(start: f32, end: f32) -> Word {
        Word { start, end, text: "x".into(), probability: 1.0, speaker: None }
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

    fn seg(start: f32, end: f32, text: &str, words: Option<Vec<Word>>) -> Seg {
        Seg { id: 0, start, end, text: text.into(), words, speaker: None }
    }

    fn spoken(start: f32, end: f32, text: &str) -> Word {
        Word { start, end, text: text.into(), probability: 1.0, speaker: None }
    }

    #[test]
    fn a_single_speaker_segment_is_not_split() {
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, "hello"),
            spoken(1.0, 2.0, "world"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[0].text, " hello world");
        let words = out[0].words.as_ref().unwrap();
        assert!(words.iter().all(|w| w.speaker == Some(1)));
    }

    #[test]
    fn a_segment_splits_where_the_speaker_changes() {
        let turns = vec![turn(0.0, 1.0, 1), turn(1.0, 2.0, 2)];
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, "hello"),
            spoken(1.0, 2.0, "world"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2, "one segment per speaker run");
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[0].text, "hello");
        assert_eq!(out[0].start, 0.0);
        assert_eq!(out[0].end, 1.0);
        assert_eq!(out[1].speaker, Some(2));
        assert_eq!(out[1].text, "world");
        assert_eq!(out[1].start, 1.0);
        assert_eq!(out[1].end, 2.0);
    }

    #[test]
    fn a_split_segments_bounds_come_from_its_own_words() {
        // Not from the original segment: the second half must not claim the
        // first half's start, or the timeline overlaps itself.
        let turns = vec![turn(0.0, 1.0, 1), turn(1.0, 3.0, 2)];
        let segs = vec![seg(0.0, 3.0, " a b c", Some(vec![
            spoken(0.0, 1.0, "a"),
            spoken(1.0, 2.0, "b"),
            spoken(2.0, 3.0, "c"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[0].end), (0.0, 1.0));
        assert_eq!((out[1].start, out[1].end), (1.0, 3.0));
        assert_eq!(out[1].text, "b c");
    }

    #[test]
    fn a_run_of_unassignable_words_becomes_its_own_segment() {
        let turns = vec![turn(0.0, 1.0, 1), turn(2.0, 3.0, 1)];
        let segs = vec![seg(0.0, 3.0, " a b c", Some(vec![
            spoken(0.0, 1.0, "a"),
            spoken(1.0, 2.0, "b"),   // gap between turns: unassignable
            spoken(2.0, 3.0, "c"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].speaker, Some(1));
        assert_eq!(out[1].speaker, None, "the gap is not attributed to anyone");
        assert_eq!(out[1].text, "b");
        assert_eq!(out[2].speaker, Some(1));
    }

    #[test]
    fn a_segment_with_no_words_passes_through_unassigned() {
        // The ASR produced a segment but no word alignment for it, so there is
        // nothing to assign per-word and nothing to split on.
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello", None)];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert!(out[0].words.is_none());
        assert_eq!(out[0].text, " hello");
    }

    #[test]
    fn a_segment_with_an_empty_word_list_passes_through() {
        let turns = vec![turn(0.0, 5.0, 1)];
        let segs = vec![seg(0.0, 2.0, " hello", Some(vec![]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert_eq!(out[0].text, " hello");
    }

    #[test]
    fn no_turns_leaves_every_segment_unassigned_and_unsplit() {
        let segs = vec![seg(0.0, 2.0, " hello world", Some(vec![
            spoken(0.0, 1.0, "hello"),
            spoken(1.0, 2.0, "world"),
        ]))];

        let out = assign(segs, &[]);

        assert_eq!(out.len(), 1);
        assert_eq!(out[0].speaker, None);
        assert_eq!(out[0].text, " hello world");
    }

    #[test]
    fn every_segment_is_processed_not_just_the_first() {
        let turns = vec![turn(0.0, 10.0, 3)];
        let segs = vec![
            seg(0.0, 1.0, " one", Some(vec![spoken(0.0, 1.0, "one")])),
            seg(1.0, 2.0, " two", Some(vec![spoken(1.0, 2.0, "two")])),
        ];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|s| s.speaker == Some(3)));
    }
    #[test]
    fn a_split_segments_text_is_readable_not_run_together() {
        // The shape the ASR really produces: ct2rs trims every word, so none
        // of them carry the leading space Whisper's segment text has.
        // Concatenating them yielded `Thisisspeakerone` in a real run of the
        // documented README example.
        let turns = vec![turn(0.0, 2.0, 0), turn(2.0, 4.0, 1)];
        let segs = vec![seg(0.0, 4.0, "this is speaker one", Some(vec![
            spoken(0.0, 1.0, "this"),
            spoken(1.0, 2.0, "is"),
            spoken(2.0, 3.0, "speaker"),
            spoken(3.0, 4.0, "one"),
        ]))];

        let out = assign(segs, &turns);

        assert_eq!(out.len(), 2, "the speaker change must split the segment");
        assert_eq!(out[0].text, "this is");
        assert_eq!(out[1].text, "speaker one");
    }

    #[test]
    fn punctuation_attaches_to_the_word_it_follows() {
        // Whisper emits punctuation as its own word. Spacing it like a word
        // ("zero , saying") is the obvious failure of a naive space-join.
        let words = vec![
            spoken(0.0, 1.0, "speaker"),
            spoken(1.0, 2.0, "zero"),
            spoken(2.0, 3.0, ","),
            spoken(3.0, 4.0, "saying"),
            spoken(4.0, 5.0, "it"),
            spoken(5.0, 6.0, "'s"),
            spoken(6.0, 7.0, "done"),
            spoken(7.0, 8.0, "."),
        ];

        assert_eq!(join_words(&words), "speaker zero, saying it's done.");
    }

    #[test]
    fn join_words_handles_the_empty_and_single_word_cases() {
        assert_eq!(join_words(&[]), "");
        assert_eq!(join_words(&[spoken(0.0, 1.0, "alone")]), "alone");
        // A leading punctuation word must not produce a leading space.
        assert_eq!(join_words(&[spoken(0.0, 1.0, "."), spoken(1.0, 2.0, "next")]), ". next");
    }

    #[test]
    fn an_exact_tie_resolves_the_same_way_every_time() {
        // Identical overlap AND identical start: without a final tie-break on
        // the speaker id this is decided by HashMap iteration order, so the
        // same audio could label the same word differently between runs.
        // Looped because a single pass can pass by luck.
        let turns = vec![turn(0.0, 2.0, 7), turn(0.0, 2.0, 3)];
        for _ in 0..64 {
            assert_eq!(speaker_for(&spoken(0.0, 2.0, "x"), &turns), Some(3));
        }
    }

}
