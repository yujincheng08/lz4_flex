//! High compression algorithm implementation.
//!
//! This module implements the LZ4 high compression algorithm using the HashTableHCU32
//! for better compression ratios at the cost of some performance.
//!
//! It includes two compression strategies:
//! - `compress_hc`: The standard high compression algorithm (levels 3-9)
//! - `compress_opt`: The optimal parsing algorithm for maximum compression (levels 10-12)

use crate::block::compress::{backtrack_match, count_same_bytes};
#[cfg(test)]
use crate::block::decompress;
use crate::block::{
    encode_sequence, handle_last_literals, CompressError, END_OFFSET, LAST_LITERALS, MAX_DISTANCE,
    MFLIMIT, MINMATCH,
};
#[cfg(not(feature = "safe-encode"))]
use crate::sink::PtrSink;
use crate::sink::Sink;
#[cfg(feature = "safe-encode")]
use crate::sink::SliceSink;
#[allow(unused_imports)]
use alloc::boxed::Box;
#[allow(unused_imports)]
use alloc::vec;
#[allow(unused_imports)]
use alloc::vec::Vec;

const HASHTABLE_SIZE_HC: usize = 1 << 15;
const MAX_DISTANCE_HC: usize = 1 << 16;

// LZ4MID constants (for levels 1-2)
const LZ4MID_HASH_LOG: usize = 15;
const LZ4MID_HASHTABLE_SIZE: usize = 1 << LZ4MID_HASH_LOG;

const MIN_MATCH: usize = 4;
const OPTIMAL_ML: usize = 32;
const ML_MASK: usize = 31;

/// Size of the optimal parsing buffer
const LZ4_OPT_NUM: usize = 1 << 12; // 4096

/// Number of trailing literals to consider after last match
const TRAILING_LITERALS: usize = 3;

/// Run mask for literal/match length encoding
const RUN_MASK: usize = 15;

/// Hash table with chain for LZ4 high compression.
///
/// Uses a two-level structure: `dict` maps each hash to the most recent input
/// position, and `chain_table` links older positions with the same hash via
/// stored deltas, forming an implicit linked list per hash bucket.
#[derive(Debug)]
pub struct HashTableHCU32 {
    /// Primary hash table: maps a 15-bit hash of each 4-byte sequence to the
    /// most recent input position (as `u32`) where that hash was seen.
    /// Fixed size of 2^15 entries, matching the output range of `hash_hc`.
    dict: Box<[u32; HASHTABLE_SIZE_HC]>,
    /// Chain table: stores a backward delta (as `u16`) at each position,
    /// pointing to the previous position with the same hash. Indexed by
    /// `pos & chain_mask()`. Dynamically sized (power of 2, up to
    /// `MAX_DISTANCE_HC`) based on input length, enabling efficient masking.
    chain_table: Box<[u16]>,
    /// Next input position to be inserted into the hash/chain tables.
    /// Positions below this have already been indexed; the compressor
    /// lazily inserts up to the current search offset on demand.
    next_to_update: usize,
    /// Maximum number of chain links to follow per search. Higher values
    /// yield better compression at the cost of more CPU time. Determined
    /// by the chosen compression level.
    max_attempts: usize,
}

/// A single LZ4 back-reference match.
///
/// Uses `u32` fields (12 bytes total vs 24 with `usize` on 64-bit) to reduce
/// stack pressure in the HC inner loop, which juggles up to 4 `Match` structs.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    /// Byte position in the input where this match starts (the "current" cursor).
    pub start: u32,
    /// Length of the match in bytes (including the mandatory 4-byte minimum).
    pub len: u32,
    /// Byte position of the earlier occurrence being referenced.
    /// The encoded offset/distance is `start - ref_pos`.
    pub ref_pos: u32,
}

impl Match {
    pub fn new() -> Self {
        Self {
            start: 0,
            len: 0,
            ref_pos: 0,
        }
    }

    #[inline(always)]
    pub fn end(&self) -> usize {
        self.start as usize + self.len as usize
    }

    pub fn fix(&mut self, correction: usize) {
        self.start += correction as u32;
        self.ref_pos += correction as u32;
        self.len = self.len.saturating_sub(correction as u32);
    }

    #[inline(always)]
    pub fn offset(&self) -> u16 {
        self.start.wrapping_sub(self.ref_pos) as u16
    }

    pub fn encode_to<S: Sink>(&self, input: &[u8], anchor: usize, output: &mut S) {
        encode_sequence(
            &input[anchor..self.start as usize],
            output,
            self.offset(),
            self.len as usize - MIN_MATCH,
        )
    }
}

/// Count how many consecutive bytes starting at `pos` equal a single repeated
/// byte value. `pattern` is that byte broadcast to all 4 lanes of a `u32`
/// (e.g. `0xABABABAB`). Widened to `usize` internally for batch comparison.
/// Equivalent to C's LZ4HC_countPattern.
#[inline]
fn count_pattern(input: &[u8], pos: usize, limit: usize, pattern: u32) -> usize {
    let limit = limit.min(input.len());
    let mut p = pos;

    // Extend 32-bit pattern to usize for batch comparison
    let pattern: usize = if core::mem::size_of::<usize>() == 8 {
        (pattern as usize) | ((pattern as usize) << 32)
    } else {
        pattern as usize
    };

    const STEP: usize = core::mem::size_of::<usize>();
    while p + STEP <= limit {
        let v = super::compress::get_batch_arch(input, p);
        let diff = v ^ pattern;
        if diff != 0 {
            p += (diff.trailing_zeros() / 8) as usize;
            return p - pos;
        }
        p += STEP;
    }

    // Byte-by-byte tail
    let byte_val = (pattern & 0xFF) as u8; // single repeated byte
    while p < limit && input[p] == byte_val {
        p += 1;
    }

    p - pos
}

/// Count how many consecutive bytes going backward from `pos` match a single
/// repeated byte value. `pattern` is that byte broadcast to all 4 lanes of a
/// `u32`, same encoding as [`count_pattern`].
/// Equivalent to C's LZ4HC_reverseCountPattern.
#[inline]
fn reverse_count_pattern(input: &[u8], pos: usize, low_limit: usize, pattern: u32) -> usize {
    let mut p = pos;

    while p >= low_limit + 4 {
        if super::compress::get_batch(input, p - 4) != pattern {
            break;
        }
        p -= 4;
    }

    // Byte-by-byte tail using native endian byte order (matches get_batch)
    let pattern_bytes = pattern.to_ne_bytes();
    let mut byte_idx: usize = 3;
    while p > low_limit {
        if input[p - 1] != pattern_bytes[byte_idx] {
            break;
        }
        p -= 1;
        byte_idx = if byte_idx == 0 { 3 } else { byte_idx - 1 };
    }

    pos - p
}

/// Read a u32 from a position that may span the boundary between `primary` and `secondary`.
/// When `pos + 4 > primary.len()`, the remaining bytes are read from `secondary[0..]`.
#[inline]
fn read_u32_2src(primary: &[u8], pos: usize, secondary: &[u8]) -> u32 {
    let remaining = primary.len() - pos;
    if remaining >= 4 {
        super::compress::get_batch(primary, pos)
    } else {
        let mut buf = [0u8; 4];
        buf[..remaining].copy_from_slice(&primary[pos..]);
        buf[remaining..].copy_from_slice(&secondary[..4 - remaining]);
        u32::from_le_bytes(buf)
    }
}

/// Count matching bytes forward with the reference starting in `ext_dict` and
/// potentially continuing into `input[0..]` (the prefix) when ext_dict is exhausted.
/// `ref_pos` may already be past ext_dict (when the min-match check crossed the boundary).
#[inline]
fn count_forward_ext_dict(
    input: &[u8],
    cur: usize,
    ext_dict: &[u8],
    ref_pos: usize,
    match_limit: usize,
) -> usize {
    let mut cur = cur;

    if ref_pos >= ext_dict.len() {
        let prefix_pos = ref_pos - ext_dict.len();
        return count_same_bytes(input, &mut cur, input, prefix_pos, match_limit);
    }

    let matched1 = count_same_bytes(input, &mut cur, ext_dict, ref_pos, match_limit);

    if ref_pos + matched1 >= ext_dict.len() && cur < match_limit {
        matched1 + count_same_bytes(input, &mut cur, input, 0, match_limit)
    } else {
        matched1
    }
}

impl HashTableHCU32 {
    #[inline]
    pub fn new(max_attempts: usize, input_len: usize) -> Self {
        // Dict table: fixed size, hash function already bounds to this range
        let dict = vec![0u32; HASHTABLE_SIZE_HC]
            .into_boxed_slice()
            .try_into()
            .unwrap();

        // Chain table: dynamically sized based on input length
        // min(input_len, MAX_DISTANCE_HC), at least 256, must be power of 2
        let chain_size = input_len.min(MAX_DISTANCE_HC).max(256).next_power_of_two();

        Self {
            dict,
            chain_table: vec![0u16; chain_size].into_boxed_slice(),
            next_to_update: 0,
            max_attempts,
        }
    }

    /// Reset the table for reuse, re-zeroing both tables.
    /// Avoids reallocation if the existing chain table is large enough.
    #[inline]
    fn reset(&mut self, max_attempts: usize, input_len: usize) {
        let needed_chain_size = input_len.min(MAX_DISTANCE_HC).max(256).next_power_of_two();

        self.dict.fill(0);

        // Reuse chain table if big enough, otherwise reallocate
        if self.chain_table.len() >= needed_chain_size {
            self.chain_table[..needed_chain_size].fill(0);
        } else {
            self.chain_table = vec![0u16; needed_chain_size].into_boxed_slice();
        }

        self.next_to_update = 0;
        self.max_attempts = max_attempts;
    }

    /// Prepare the table for a new linked block without clearing existing entries.
    /// Ensures the chain table is `MAX_DISTANCE_HC` so cross-block chain links work,
    /// and advances `next_to_update` past the positions now in `ext_dict`.
    fn prepare_linked_block(&mut self, max_attempts: usize, abs_block_start: usize) {
        if self.chain_table.len() < MAX_DISTANCE_HC {
            let mut new_chain = vec![0u16; MAX_DISTANCE_HC].into_boxed_slice();
            let old_len = self.chain_table.len();
            for i in 0..old_len {
                new_chain[i] = self.chain_table[i];
            }
            self.chain_table = new_chain;
        }
        self.next_to_update = abs_block_start;
        self.max_attempts = max_attempts;
    }

    /// Subtract `delta` from every absolute position stored in the hash table.
    /// Used when `stream_offset` approaches `u32::MAX / 2` to prevent overflow.
    fn reposition(&mut self, delta: usize) {
        let delta32 = delta as u32;
        for entry in self.dict.iter_mut() {
            *entry = entry.saturating_sub(delta32);
        }
        self.next_to_update = self.next_to_update.saturating_sub(delta);
    }

    /// Mask for chain table indexing (table size is always power of 2)
    #[inline]
    fn chain_mask(&self) -> usize {
        self.chain_table.len() - 1
    }

    /// Get the next position in the chain for a given offset
    #[inline(always)]
    fn next(&self, pos: usize) -> usize {
        let idx = pos & self.chain_mask();
        pos - (self.chain_table[idx] as usize)
    }

    /// Get the raw chain delta at a position (equivalent to C's DELTANEXTU16)
    #[inline(always)]
    fn chain_delta(&self, pos: usize) -> u16 {
        let idx = pos & self.chain_mask();
        self.chain_table[idx]
    }

    #[inline(always)]
    fn add_hash(&mut self, hash: usize, pos: usize) {
        let chain_idx = pos & self.chain_mask();
        let delta = pos - self.dict[hash] as usize;
        let delta = if delta > self.chain_mask() {
            self.chain_mask()
        } else {
            delta
        };
        self.chain_table[chain_idx] = delta as u16;
        self.dict[hash] = pos as u32;
    }

    /// Get dict value at hash position
    #[inline(always)]
    fn get_dict(&self, hash: usize) -> usize {
        self.dict[hash] as usize
    }

    /// Set dict value at hash position
    #[inline(always)]
    fn set_dict(&mut self, hash: usize, pos: usize) {
        self.dict[hash] = pos as u32;
    }

    /// Set chain value at position
    #[inline(always)]
    fn set_chain(&mut self, pos: usize, delta: u16) {
        let idx = pos & self.chain_mask();
        self.chain_table[idx] = delta;
    }

    /// Hash function for high compression
    #[inline]
    fn hash_hc(v: u32) -> u32 {
        v.wrapping_mul(2654435761u32) >> 17
    }

    #[inline]
    fn get_hash_at(input: &[u8], pos: usize) -> usize {
        Self::hash_hc(super::compress::get_batch(input, pos)) as usize
    }

    /// Insert hashes for all positions up to the given local offset.
    /// Positions stored in the hash table are absolute (`local_pos + stream_offset`).
    #[inline]
    pub fn insert(&mut self, off: u32, input: &[u8], stream_offset: usize) {
        let abs_off = off as usize + stream_offset;
        for abs_pos in self.next_to_update..abs_off {
            let local_pos = abs_pos - stream_offset;
            self.add_hash(Self::get_hash_at(input, local_pos), abs_pos);
        }
        self.next_to_update = abs_off;
    }

    fn insert_and_find_best_match(
        &mut self,
        input: &[u8],
        off: u32,
        match_limit: u32,
        match_info: &mut Match,
        ext_dict: &[u8],
        stream_offset: usize,
    ) -> bool {
        match_info.start = off;
        match_info.len = 0;
        let mut delta: usize = 0;
        let mut repl: usize = 0;

        let off = off as usize;
        let match_limit = match_limit as usize;
        let abs_off = off + stream_offset;
        let ext_dict_stream_offset = stream_offset - ext_dict.len();

        self.insert(off as u32, input, stream_offset);

        let mut candidate_abs = self.get_dict(Self::get_hash_at(input, off));

        for i in 0..self.max_attempts {
            if candidate_abs >= abs_off || abs_off - candidate_abs > self.chain_mask() {
                break;
            }

            if candidate_abs >= stream_offset {
                let ref_local = candidate_abs - stream_offset;

                if match_info.len >= MIN_MATCH as u32 {
                    let check_pos = match_info.len as usize - 1;
                    if input[ref_local + check_pos] != input[off + check_pos]
                        || input[ref_local + check_pos + 1] != input[off + check_pos + 1]
                    {
                        let next = self.next(candidate_abs);
                        if next >= abs_off
                            || abs_off - next > self.chain_mask()
                            || next == candidate_abs
                        {
                            break;
                        }
                        candidate_abs = next;
                        continue;
                    }
                }

                if self.read_min_match_equals(input, ref_local, off) {
                    let match_len = MIN_MATCH
                        + self.common_bytes(
                            input,
                            ref_local + MIN_MATCH,
                            off + MIN_MATCH,
                            match_limit,
                        );
                    if match_len as u32 > match_info.len {
                        let distance = abs_off - candidate_abs;
                        match_info.ref_pos = (off as u32).wrapping_sub(distance as u32);
                        match_info.len = match_len as u32;
                    }
                    if i == 0 {
                        repl = match_len;
                        delta = abs_off - candidate_abs;
                    }
                }
            } else if !ext_dict.is_empty() && candidate_abs >= ext_dict_stream_offset {
                let ref_local = candidate_abs - ext_dict_stream_offset;

                if ref_local + 4 <= ext_dict.len() {
                    if super::compress::get_batch(ext_dict, ref_local)
                        == super::compress::get_batch(input, off)
                    {
                        let match_len = MIN_MATCH
                            + count_forward_ext_dict(
                                input,
                                off + MIN_MATCH,
                                ext_dict,
                                ref_local + MIN_MATCH,
                                match_limit,
                            );
                        if match_len as u32 > match_info.len {
                            let distance = abs_off - candidate_abs;
                            match_info.ref_pos = (off as u32).wrapping_sub(distance as u32);
                            match_info.len = match_len as u32;
                        }
                        if i == 0 {
                            repl = match_len;
                            delta = abs_off - candidate_abs;
                        }
                    }
                } else if ref_local < ext_dict.len() {
                    if read_u32_2src(ext_dict, ref_local, input)
                        == super::compress::get_batch(input, off)
                    {
                        let match_len = MIN_MATCH
                            + count_forward_ext_dict(
                                input,
                                off + MIN_MATCH,
                                ext_dict,
                                ref_local + MIN_MATCH,
                                match_limit,
                            );
                        if match_len as u32 > match_info.len {
                            let distance = abs_off - candidate_abs;
                            match_info.ref_pos = (off as u32).wrapping_sub(distance as u32);
                            match_info.len = match_len as u32;
                        }
                        if i == 0 {
                            repl = match_len;
                            delta = abs_off - candidate_abs;
                        }
                    }
                }
            }

            let next = self.next(candidate_abs);
            if next >= abs_off || abs_off - next > self.chain_mask() || next == candidate_abs {
                break;
            }
            candidate_abs = next;
        }

        // Handle pre hash (positions are absolute for hash table, local for input reads)
        if repl != 0 {
            let mut abs_ptr = abs_off;
            let abs_end = abs_off + repl - 3;
            while abs_ptr < abs_end - delta {
                self.set_chain(abs_ptr, delta as u16);
                abs_ptr += 1;
            }
            loop {
                self.set_chain(abs_ptr, delta as u16);
                let local_ptr = abs_ptr - stream_offset;
                self.set_dict(Self::get_hash_at(input, local_ptr), abs_ptr);
                abs_ptr += 1;
                if abs_ptr >= abs_end {
                    break;
                }
            }
            self.next_to_update = abs_end;
        }

        match_info.len != 0
    }

    /// Insert hashes and find a wider match, similar to Java insertAndFindWiderMatch
    pub fn insert_and_find_wider_match(
        &mut self,
        input: &[u8],
        off: u32,
        start_limit: u32,
        match_limit: u32,
        min_len: u32,
        match_info: &mut Match,
        ext_dict: &[u8],
        stream_offset: usize,
    ) -> bool {
        match_info.len = min_len;

        let off = off as usize;
        let start_limit = start_limit as usize;
        let match_limit = match_limit as usize;
        let abs_off = off + stream_offset;
        let ext_dict_stream_offset = stream_offset - ext_dict.len();

        let look_back_length = off - start_limit;

        self.insert(off as u32, input, stream_offset);

        let mut candidate_abs = self.get_dict(Self::get_hash_at(input, off));

        for _ in 0..self.max_attempts {
            if candidate_abs >= abs_off || abs_off - candidate_abs > self.chain_mask() {
                break;
            }

            if candidate_abs >= stream_offset {
                let ref_local = candidate_abs - stream_offset;

                if match_info.len >= MIN_MATCH as u32 && ref_local >= look_back_length {
                    let src_check = start_limit + match_info.len as usize - 1;
                    let match_check = ref_local - look_back_length + match_info.len as usize - 1;
                    if input[src_check] != input[match_check]
                        || input[src_check + 1] != input[match_check + 1]
                    {
                        let next = self.next(candidate_abs);
                        if next >= abs_off
                            || abs_off - next > self.chain_mask()
                            || next == candidate_abs
                        {
                            break;
                        }
                        candidate_abs = next;
                        continue;
                    }
                }

                if self.read_min_match_equals(input, ref_local, off) {
                    let match_len_forward = MIN_MATCH
                        + self.common_bytes(
                            input,
                            ref_local + MIN_MATCH,
                            off + MIN_MATCH,
                            match_limit,
                        );
                    let match_len_backward =
                        Self::common_bytes_backward(input, ref_local, off, 0, start_limit);
                    let match_len = (match_len_backward + match_len_forward) as u32;

                    if match_len > match_info.len {
                        match_info.len = match_len;
                        let distance = abs_off - candidate_abs;
                        match_info.ref_pos =
                            ((off - match_len_backward) as u32).wrapping_sub(distance as u32);
                        match_info.start = (off - match_len_backward) as u32;
                    }
                }
            } else if !ext_dict.is_empty() && candidate_abs >= ext_dict_stream_offset {
                let ref_local = candidate_abs - ext_dict_stream_offset;

                let min_match_ok = if ref_local + 4 <= ext_dict.len() {
                    super::compress::get_batch(ext_dict, ref_local)
                        == super::compress::get_batch(input, off)
                } else if ref_local < ext_dict.len() {
                    read_u32_2src(ext_dict, ref_local, input)
                        == super::compress::get_batch(input, off)
                } else {
                    false
                };

                if min_match_ok {
                    let match_len_forward = MIN_MATCH
                        + count_forward_ext_dict(
                            input,
                            off + MIN_MATCH,
                            ext_dict,
                            ref_local + MIN_MATCH,
                            match_limit,
                        );
                    // No backward extension for ext_dict matches
                    let match_len = match_len_forward as u32;

                    if match_len > match_info.len {
                        match_info.len = match_len;
                        let distance = abs_off - candidate_abs;
                        match_info.ref_pos = (off as u32).wrapping_sub(distance as u32);
                        match_info.start = off as u32;
                    }
                }
            }

            let next = self.next(candidate_abs);
            if next >= abs_off || abs_off - next > self.chain_mask() || next == candidate_abs {
                break;
            }
            candidate_abs = next;
        }

        match_info.len > min_len
    }

    /// Check if two 4-byte sequences starting at the given positions are equal
    #[inline]
    fn read_min_match_equals(&self, input: &[u8], pos1: usize, pos2: usize) -> bool {
        // Fast u32 comparison instead of slice comparison
        super::compress::get_batch(input, pos1) == super::compress::get_batch(input, pos2)
    }

    /// Find the number of common bytes between two positions (optimized version)
    /// Count matching bytes forward. Delegates to the shared `count_same_bytes`
    /// with `input` as both slices (HC always matches within the same buffer).
    #[inline]
    fn common_bytes(&self, input: &[u8], pos1: usize, pos2: usize, limit: usize) -> usize {
        let mut cur = pos2;
        count_same_bytes(input, &mut cur, input, pos1, input.len().min(limit))
    }

    /// Find the number of common bytes backward from two positions (optimized)
    #[inline]
    fn common_bytes_backward(
        input: &[u8],
        mut pos1: usize,
        mut pos2: usize,
        limit1: usize,
        limit2: usize,
    ) -> usize {
        let mut len = 0;
        let max_back = (pos1 - limit1).min(pos2 - limit2);

        if max_back == 0 {
            return 0;
        }

        // Process usize (8 bytes on 64-bit) at a time, backwards
        const STEP_SIZE: usize = core::mem::size_of::<usize>();
        while len + STEP_SIZE <= max_back {
            let v1 = super::compress::get_batch_arch(input, pos1 - len - STEP_SIZE);
            let v2 = super::compress::get_batch_arch(input, pos2 - len - STEP_SIZE);
            let diff = v1 ^ v2;

            if diff == 0 {
                len += STEP_SIZE;
            } else {
                // Find first differing byte from the end (using leading zeros for backward)
                return len + (diff.to_be().trailing_zeros() / 8) as usize;
            }
        }

        // Update positions to account for bytes already compared in batch loop
        pos1 -= len;
        pos2 -= len;

        // Process remaining 4 bytes if on 64-bit
        #[cfg(target_pointer_width = "64")]
        if len + 4 <= max_back {
            let v1 = super::compress::get_batch(input, pos1 - 4);
            let v2 = super::compress::get_batch(input, pos2 - 4);
            let diff = v1 ^ v2;
            if diff == 0 {
                len += 4;
                pos1 -= 4;
                pos2 -= 4;
            } else {
                return len + (diff.to_be().trailing_zeros() / 8) as usize;
            }
        }

        // Process remaining 2 bytes
        if len + 2 <= max_back {
            if input[pos1 - 2] == input[pos2 - 2] && input[pos1 - 1] == input[pos2 - 1] {
                len += 2;
                pos1 -= 2;
                pos2 -= 2;
            } else if input[pos1 - 1] == input[pos2 - 1] {
                return len + 1;
            } else {
                return len;
            }
        }

        // Process last byte
        if len < max_back && input[pos1 - 1] == input[pos2 - 1] {
            len += 1;
        }

        len
    }

    /// Find the longest match at `off`, returning `(match_len_u32, offset_u16)`.
    /// Uses u32 params to reduce call overhead (LZ4 block max is ~2GB).
    /// Offset is u16 since LZ4 format limits distance to 16 bits.
    #[inline]
    pub fn find_longer_match(
        &mut self,
        input: &[u8],
        off: u32,
        match_limit: u32,
        min_len: u32,
        ext_dict: &[u8],
        stream_offset: usize,
    ) -> (u32, u16) {
        self.insert(off, input, stream_offset);

        let off = off as usize;
        let match_limit = match_limit as usize;
        let abs_off = off + stream_offset;
        let ext_dict_stream_offset = stream_offset - ext_dict.len();

        let mut best_len: usize = min_len as usize;
        let mut best_offset: u16 = 0;
        let mut match_chain_pos: usize = 0;

        let mut repeat: u8 = 0;
        let mut src_pattern_length: usize = 0;

        let mut candidate_abs = self.get_dict(Self::get_hash_at(input, off));

        for _ in 0..self.max_attempts {
            if candidate_abs >= abs_off || abs_off - candidate_abs > self.chain_mask() {
                break;
            }

            let mut match_len: usize = 0;
            let ref_in_input = candidate_abs >= stream_offset;

            if ref_in_input {
                let ref_local = candidate_abs - stream_offset;

                let pre_check_ok = if best_len >= MIN_MATCH {
                    let check_pos = best_len - 1;
                    #[cfg(not(feature = "safe-encode"))]
                    unsafe {
                        (input.as_ptr().add(ref_local + check_pos) as *const u16).read_unaligned()
                            == (input.as_ptr().add(off + check_pos) as *const u16).read_unaligned()
                    }
                    #[cfg(feature = "safe-encode")]
                    {
                        input[ref_local + check_pos] == input[off + check_pos]
                            && input[ref_local + check_pos + 1] == input[off + check_pos + 1]
                    }
                } else {
                    true
                };

                if pre_check_ok && self.read_min_match_equals(input, ref_local, off) {
                    match_len = MIN_MATCH
                        + self.common_bytes(
                            input,
                            ref_local + MIN_MATCH,
                            off + MIN_MATCH,
                            match_limit,
                        );
                    if match_len > best_len {
                        best_len = match_len;
                        best_offset = (abs_off - candidate_abs) as u16;
                    }
                }

                // Chain swap: only for input matches
                if match_len == best_len
                    && match_len >= MIN_MATCH
                    && candidate_abs + best_len <= abs_off
                {
                    const K_TRIGGER: i32 = 4;
                    let mut dist_to_next: u16 = 1;
                    let end = (best_len - MIN_MATCH + 1) as i32;
                    let mut accel: i32 = 1 << K_TRIGGER;
                    let mut pos: i32 = 0;
                    while pos < end {
                        let candidate_dist =
                            self.chain_delta(candidate_abs.wrapping_add(pos as usize));
                        let step = accel >> K_TRIGGER;
                        accel += 1;
                        if candidate_dist > dist_to_next {
                            dist_to_next = candidate_dist;
                            match_chain_pos = pos as usize;
                            accel = 1 << K_TRIGGER;
                        }
                        pos += step;
                    }
                    if dist_to_next > 1 {
                        if (dist_to_next as usize) > candidate_abs {
                            break;
                        }
                        candidate_abs -= dist_to_next as usize;
                        continue;
                    }
                }

                // Pattern analysis: only for input matches
                {
                    let dist_next = self.chain_delta(candidate_abs);
                    if dist_next == 1 && match_chain_pos == 0 {
                        let match_candidate = candidate_abs.wrapping_sub(1);
                        if repeat == 0 {
                            let pattern = super::compress::get_batch(input, off);
                            if (pattern & 0xFFFF) == (pattern >> 16)
                                && (pattern & 0xFF) == (pattern >> 24)
                            {
                                repeat = 1;
                                src_pattern_length =
                                    count_pattern(input, off + 4, match_limit, pattern) + 4;
                            } else {
                                repeat = 2;
                            }
                        }
                        if repeat == 1
                            && match_candidate < abs_off
                            && abs_off - match_candidate <= self.chain_mask()
                            && match_candidate >= stream_offset
                        {
                            let mc_local = match_candidate - stream_offset;
                            let pattern = super::compress::get_batch(input, off);
                            if mc_local + 4 <= input.len()
                                && super::compress::get_batch(input, mc_local) == pattern
                            {
                                let forward_pattern_len =
                                    count_pattern(input, mc_local + 4, match_limit, pattern) + 4;
                                let back_length =
                                    reverse_count_pattern(input, mc_local, 0, pattern);
                                let current_segment_len = back_length + forward_pattern_len;

                                if current_segment_len >= src_pattern_length
                                    && forward_pattern_len <= src_pattern_length
                                {
                                    let new_ref_local =
                                        mc_local + forward_pattern_len - src_pattern_length;
                                    let new_ref_abs = new_ref_local + stream_offset;
                                    if abs_off > new_ref_abs
                                        && abs_off - new_ref_abs <= self.chain_mask()
                                    {
                                        candidate_abs = new_ref_abs;
                                        continue;
                                    }
                                } else {
                                    let new_ref_local = mc_local - back_length;
                                    let new_ref_abs = new_ref_local + stream_offset;
                                    if abs_off > new_ref_abs
                                        && abs_off - new_ref_abs <= self.chain_mask()
                                    {
                                        let max_ml = current_segment_len.min(src_pattern_length);
                                        if max_ml > best_len {
                                            best_len = max_ml;
                                            best_offset = (abs_off - new_ref_abs) as u16;
                                        }
                                        let dist = self.chain_delta(new_ref_abs) as usize;
                                        if dist == 0 || dist > new_ref_abs {
                                            break;
                                        }
                                        candidate_abs = new_ref_abs - dist;
                                        continue;
                                    }
                                }
                            }
                        }
                    }
                }
            } else if !ext_dict.is_empty() && candidate_abs >= ext_dict_stream_offset {
                let ref_local = candidate_abs - ext_dict_stream_offset;

                let min_match_ok = if ref_local + 4 <= ext_dict.len() {
                    super::compress::get_batch(ext_dict, ref_local)
                        == super::compress::get_batch(input, off)
                } else if ref_local < ext_dict.len() {
                    read_u32_2src(ext_dict, ref_local, input)
                        == super::compress::get_batch(input, off)
                } else {
                    false
                };

                if min_match_ok {
                    match_len = MIN_MATCH
                        + count_forward_ext_dict(
                            input,
                            off + MIN_MATCH,
                            ext_dict,
                            ref_local + MIN_MATCH,
                            match_limit,
                        );
                    if match_len > best_len {
                        best_len = match_len;
                        best_offset = (abs_off - candidate_abs) as u16;
                    }
                }
                // Skip chain swap and pattern analysis for ext_dict matches
            }

            let delta = self.chain_delta(candidate_abs + match_chain_pos) as usize;
            if delta == 0 || delta > candidate_abs {
                break;
            }
            candidate_abs -= delta;
        }

        if best_len > min_len as usize {
            (best_len as u32, best_offset)
        } else {
            (0, 0)
        }
    }
}

/// Optimal parsing state for a single position.
/// Matches C's LZ4HC_optimal_t layout (4x i32 = 16 bytes).
/// Using i32 for off/mlen instead of u16 avoids costly widening conversions
/// on every access in the hot DP loop (15-20% regression with u16).
/// The 4099-entry opt array is ~64KB.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct OptimalState {
    /// Cost in bytes to reach this position
    price: i32,
    /// Literal length before this position
    litlen: i32,
    /// Match offset (0 for literal)
    off: i32,
    /// Match length (1 for literal)
    mlen: i32,
}

impl OptimalState {
    const SENTINEL: Self = Self {
        price: i32::MAX,
        litlen: 0,
        off: 0,
        mlen: 0,
    };
}

/// Calculate the cost in bytes of encoding literals
#[inline]
fn literals_price(litlen: i32) -> i32 {
    let mut price = litlen;
    if litlen >= RUN_MASK as i32 {
        price += 1 + (litlen - RUN_MASK as i32) / 255;
    }
    price
}

/// Calculate the cost in bytes of encoding a sequence (literals + match)
#[inline]
fn sequence_price(litlen: i32, mlen: i32) -> i32 {
    // token + 16-bit offset
    let mut price: i32 = 1 + 2;

    // literal length encoding
    price += literals_price(litlen);

    // match length encoding (mlen >= MINMATCH)
    let ml_code = mlen - MIN_MATCH as i32;
    if ml_code >= 15 {
        price += 1 + (ml_code - 15) / 255;
    }

    price
}

/// A reusable compression table for the HC algorithm that avoids re-allocating
/// internal hash tables on every call.
///
/// This is useful when compressing many inputs in a loop (e.g. frame blocks).
/// Create one table and pass it to [`compress_hc_with_table`] repeatedly.
///
/// The table automatically selects the right internal variant (mid vs HC)
/// based on the compression level, upgrading transparently when needed.
///
/// # Example
/// ```
/// use lz4_flex::block::{compress_hc_to_vec_with_table, CompressTableHC};
///
/// let mut table = CompressTableHC::new();
/// for input in [b"block one".as_slice(), b"block two".as_slice()] {
///     let compressed = compress_hc_to_vec_with_table(input, 9, &mut table);
/// }
/// ```
pub struct CompressTableHC {
    inner: CompressTableHCInner,
    compression_level: u8,
}

enum CompressTableHCInner {
    Mid(HashTableMid),
    HC(HashTableHCU32),
}

impl CompressTableHC {
    /// Create a new table. The internal variant is lazily chosen on first use.
    pub fn new() -> Self {
        CompressTableHC {
            inner: CompressTableHCInner::Mid(HashTableMid::new()),
            compression_level: 0,
        }
    }

    /// Prepare the table for a new linked block without clearing existing entries.
    /// Called by `FrameEncoder` between blocks in linked mode.
    pub(crate) fn prepare_linked_block(&mut self, level: u8, abs_block_start: usize) {
        let level = level.min(12);
        let max_attempts = if level >= 10 {
            match level {
                10 => 96,
                11 => 512,
                _ => 16384,
            }
        } else if level >= 3 {
            1 << (level - 1)
        } else {
            0
        };

        if level >= 3 {
            match &mut self.inner {
                CompressTableHCInner::HC(ht) => {
                    ht.prepare_linked_block(max_attempts, abs_block_start);
                }
                _ => {
                    let mut ht = HashTableHCU32::new(max_attempts, MAX_DISTANCE_HC);
                    ht.prepare_linked_block(max_attempts, abs_block_start);
                    self.inner = CompressTableHCInner::HC(ht);
                }
            }
        } else {
            match &mut self.inner {
                CompressTableHCInner::Mid(mid) => {
                    mid.prepare_linked_block();
                }
                _ => {
                    self.inner = CompressTableHCInner::Mid(HashTableMid::new());
                }
            }
        }
        self.compression_level = level;
    }

    /// Subtract `delta` from all stored positions to prevent overflow.
    pub(crate) fn reposition(&mut self, delta: usize) {
        match &mut self.inner {
            CompressTableHCInner::HC(ht) => ht.reposition(delta),
            CompressTableHCInner::Mid(mid) => mid.reposition(delta),
        }
    }
}

/// Compress input data using the LZ4 high compression algorithm.
///
/// Allocates a fresh internal table on every call. For repeated compression
/// (e.g. compressing many blocks in a loop), prefer [`compress_hc_with_table`]
/// with a reusable [`CompressTableHC`] to avoid repeated allocation.
///
/// # Compression levels
/// - **Levels 1-2**: lz4mid intermediate algorithm
/// - **Levels 3-9**: HC hash chain algorithm with increasing search depth
/// - **Levels 10-12**: Optimal parsing (dynamic programming) for maximum compression
///
/// # Example
/// ```ignore
/// use lz4_flex::block::compress_hc;
/// let input = b"Hello, this is some data to compress!";
/// let mut output = vec![0u8; input.len() * 2];
/// let size = compress_hc(input, &mut output, 9).unwrap(); // HC algorithm
/// let size = compress_hc(input, &mut output, 12).unwrap(); // Optimal algorithm
/// ```
pub fn compress_hc(
    input: &[u8],
    output: &mut impl Sink,
    level: u8,
) -> Result<usize, CompressError> {
    let level = level.min(12);

    if level >= 10 {
        let nb_searches = match level {
            10 => 96,
            11 => 512,
            _ => 16384,
        };
        let mut ht = HashTableHCU32::new(nb_searches, input.len());
        compress_opt_internal(input, 0, output, level, &mut ht, &[], 0)
    } else if level >= 3 {
        let mut ht = HashTableHCU32::new(1 << (level - 1), input.len());
        compress_hc_internal(input, 0, output, &mut ht, &[], 0)
    } else {
        let mut table = HashTableMid::new();
        compress_mid_internal(input, 0, output, &mut table, &[], 0)
    }
}

/// Compress input data using the LZ4 high compression algorithm, reusing a
/// [`CompressTableHC`] to avoid re-allocating internal hash tables.
///
/// The table is automatically reset before each call. If the level changes
/// between calls (e.g. mid vs HC), the table is transparently upgraded.
///
/// See [`compress_hc`] for compression level details.
///
/// # Example
/// ```ignore
/// use lz4_flex::block::{compress_hc_with_table, get_maximum_output_size, CompressTableHC};
///
/// let mut table = CompressTableHC::new();
/// let input = b"Hello, this is some data to compress with HC!";
/// let mut output = vec![0u8; get_maximum_output_size(input.len())];
/// let n = compress_hc_with_table(input, &mut output, 9, &mut table).unwrap();
/// ```
pub fn compress_hc_with_table(
    input: &[u8],
    output: &mut impl Sink,
    level: u8,
    table: &mut CompressTableHC,
) -> Result<usize, CompressError> {
    let level = level.min(12);

    if level >= 3 {
        let max_attempts = if level >= 10 {
            match level {
                10 => 96,
                11 => 512,
                _ => 16384,
            }
        } else {
            1 << (level - 1)
        };

        let ht = match &mut table.inner {
            CompressTableHCInner::HC(ht) => {
                ht.reset(max_attempts, input.len());
                ht
            }
            _ => {
                table.inner =
                    CompressTableHCInner::HC(HashTableHCU32::new(max_attempts, input.len()));
                match &mut table.inner {
                    CompressTableHCInner::HC(ht) => ht,
                    _ => unreachable!(),
                }
            }
        };

        if level >= 10 {
            compress_opt_internal(input, 0, output, level, ht, &[], 0)
        } else {
            compress_hc_internal(input, 0, output, ht, &[], 0)
        }
    } else {
        let mid = match &mut table.inner {
            CompressTableHCInner::Mid(mid) => {
                mid.reset();
                mid
            }
            _ => {
                table.inner = CompressTableHCInner::Mid(HashTableMid::new());
                match &mut table.inner {
                    CompressTableHCInner::Mid(mid) => mid,
                    _ => unreachable!(),
                }
            }
        };
        compress_mid_internal(input, 0, output, mid, &[], 0)
    }
}

/// Compress with HC using linked-block mode. Called by the frame encoder.
/// `input` includes the prefix at `[0..input_pos]`, current block at `[input_pos..]`.
/// `table` must have been prepared via `prepare_linked_block` for the current block.
pub(crate) fn compress_hc_linked(
    input: &[u8],
    input_pos: usize,
    output: &mut impl Sink,
    level: u8,
    table: &mut CompressTableHC,
    ext_dict: &[u8],
    stream_offset: usize,
) -> Result<usize, CompressError> {
    let level = level.min(12);

    if level >= 3 {
        let ht = match &mut table.inner {
            CompressTableHCInner::HC(ht) => ht,
            _ => unreachable!("prepare_linked_block should have ensured HC variant for level >= 3"),
        };
        if level >= 10 {
            compress_opt_internal(input, input_pos, output, level, ht, ext_dict, stream_offset)
        } else {
            compress_hc_internal(input, input_pos, output, ht, ext_dict, stream_offset)
        }
    } else {
        let mid = match &mut table.inner {
            CompressTableHCInner::Mid(mid) => mid,
            _ => unreachable!("prepare_linked_block should have ensured Mid variant for level < 3"),
        };
        compress_mid_internal(input, input_pos, output, mid, ext_dict, stream_offset)
    }
}

/// Compress input data using the LZ4 high compression algorithm, returning a Vec.
///
/// This is a convenience function that allocates the output buffer internally.
/// See [`compress_hc`] for details on the compression algorithm and levels.
///
/// # Arguments
/// * `input` - The input data to compress
/// * `level` - Compression level (1-12), higher means better compression but slower
///
/// # Returns
/// A Vec containing the compressed data
///
/// # Example
/// ```
/// use lz4_flex::block::compress_hc_to_vec;
/// let input = b"Hello, this is some data to compress!";
/// let compressed = compress_hc_to_vec(input, 9); // HC algorithm
/// let compressed = compress_hc_to_vec(input, 12); // Optimal algorithm
/// ```
pub fn compress_hc_to_vec(input: &[u8], level: u8) -> Vec<u8> {
    let max_size = crate::block::compress::get_maximum_output_size(input.len());
    #[cfg(feature = "safe-encode")]
    {
        let mut output = vec![0u8; max_size];
        let mut sink = SliceSink::new(&mut output, 0);
        let compressed_size = compress_hc(input, &mut sink, level).unwrap();
        output.truncate(compressed_size);
        output
    }
    #[cfg(not(feature = "safe-encode"))]
    {
        let mut output = Vec::with_capacity(max_size);
        let compressed_size =
            compress_hc(input, &mut PtrSink::from_vec(&mut output, 0), level).unwrap();
        unsafe {
            output.set_len(compressed_size);
        }
        output.shrink_to_fit();
        output
    }
}

/// Compress input data using the LZ4 high compression algorithm, returning a Vec
/// and reusing a [`CompressTableHC`].
///
/// This is the reusable-table variant of [`compress_hc_to_vec`]. See
/// [`compress_hc_with_table`] for details on table reuse.
///
/// # Example
/// ```
/// use lz4_flex::block::{compress_hc_to_vec_with_table, CompressTableHC};
///
/// let mut table = CompressTableHC::new();
/// let compressed = compress_hc_to_vec_with_table(b"data to compress", 9, &mut table);
/// ```
pub fn compress_hc_to_vec_with_table(
    input: &[u8],
    level: u8,
    table: &mut CompressTableHC,
) -> Vec<u8> {
    let max_size = crate::block::compress::get_maximum_output_size(input.len());
    #[cfg(feature = "safe-encode")]
    {
        let mut output = vec![0u8; max_size];
        let mut sink = SliceSink::new(&mut output, 0);
        let compressed_size = compress_hc_with_table(input, &mut sink, level, table).unwrap();
        output.truncate(compressed_size);
        output
    }
    #[cfg(not(feature = "safe-encode"))]
    {
        let mut output = Vec::with_capacity(max_size);
        let compressed_size =
            compress_hc_with_table(input, &mut PtrSink::from_vec(&mut output, 0), level, table)
                .unwrap();
        unsafe {
            output.set_len(compressed_size);
        }
        output.shrink_to_fit();
        output
    }
}

// ============================================================================
// LZ4MID - Intermediate compression (levels 1-2)
// Uses two hash tables (4-byte and 8-byte) for better compression than fast
// algorithm while being faster than HC.
// ============================================================================

/// Hash table for lz4mid algorithm - contains two tables (4-byte and 8-byte)
pub(crate) struct HashTableMid {
    hash4: Box<[u32; LZ4MID_HASHTABLE_SIZE]>,
    hash8: Box<[u32; LZ4MID_HASHTABLE_SIZE]>,
}

impl HashTableMid {
    fn new() -> Self {
        HashTableMid {
            hash4: vec![0u32; LZ4MID_HASHTABLE_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            hash8: vec![0u32; LZ4MID_HASHTABLE_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    /// Reset the table for reuse by zeroing both hash tables.
    fn reset(&mut self) {
        self.hash4.fill(0);
        self.hash8.fill(0);
    }

    /// Prepare the table for a new linked block without clearing entries.
    fn prepare_linked_block(&mut self) {
        // Hash tables persist; no action needed since entries are absolute positions.
    }

    /// Subtract `delta` from every absolute position stored in both hash tables.
    fn reposition(&mut self, delta: usize) {
        let delta32 = delta as u32;
        for entry in self.hash4.iter_mut() {
            *entry = entry.saturating_sub(delta32);
        }
        for entry in self.hash8.iter_mut() {
            *entry = entry.saturating_sub(delta32);
        }
    }
}

/// 4-byte hash for lz4mid (same multiplier as fast algorithm)
#[inline]
fn get_hash4_mid(input: &[u8], pos: usize) -> usize {
    let v = super::compress::get_batch(input, pos);
    (v.wrapping_mul(2654435761) >> (32 - LZ4MID_HASH_LOG)) as usize
}

/// 8-byte hash for lz4mid (hashes lower 56 bits for longer match detection)
#[inline]
fn get_hash8_mid(input: &[u8], pos: usize) -> usize {
    // Use get_batch_arch for the raw read (eliminates bounds check in unsafe mode),
    // then convert to u64 for the 56-bit hash computation.
    #[cfg(target_pointer_width = "64")]
    {
        let v = super::compress::get_batch_arch(input, pos) as u64;
        let v56 = v.to_le() << 8;
        ((v56.wrapping_mul(58295818150454627)) >> (64 - LZ4MID_HASH_LOG)) as usize
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        let v = u64::from_le_bytes(input[pos..pos + 8].try_into().unwrap());
        let v56 = v << 8;
        ((v56.wrapping_mul(58295818150454627)) >> (64 - LZ4MID_HASH_LOG)) as usize
    }
}

/// Internal lz4mid compression.
/// `input_pos` is where the current block starts (positions before it are prefix).
/// `ext_dict` and `stream_offset` support linked block mode.
fn compress_mid_internal(
    input: &[u8],
    input_pos: usize,
    output: &mut impl Sink,
    table: &mut HashTableMid,
    ext_dict: &[u8],
    stream_offset: usize,
) -> Result<usize, CompressError> {
    let output_start = output.pos();

    if input.len() - input_pos < MFLIMIT + 1 {
        handle_last_literals(output, &input[input_pos..]);
        return Ok(output.pos() - output_start);
    }

    let hash4 = &mut *table.hash4;
    let hash8 = &mut *table.hash8;

    let ext_dict_stream_offset = stream_offset - ext_dict.len();

    let mut ip = input_pos;
    let mut anchor = input_pos;
    let input_end = input.len();
    let mflimit = input_end.saturating_sub(MFLIMIT);
    let ilimit = input_end.saturating_sub(8);
    let match_limit = input_end - END_OFFSET;

    #[inline(always)]
    fn add_hash8(
        hash8: &mut [u32; LZ4MID_HASHTABLE_SIZE],
        input: &[u8],
        pos: usize,
        input_end: usize,
        stream_offset: usize,
    ) {
        if pos + 8 <= input_end {
            let h = get_hash8_mid(input, pos);
            hash8[h] = (pos + stream_offset) as u32;
        }
    }

    #[inline(always)]
    fn add_hash4(
        hash4: &mut [u32; LZ4MID_HASHTABLE_SIZE],
        input: &[u8],
        pos: usize,
        input_end: usize,
        stream_offset: usize,
    ) {
        if pos + 4 <= input_end {
            let h = get_hash4_mid(input, pos);
            hash4[h] = (pos + stream_offset) as u32;
        }
    }

    /// Resolve an absolute hash table position to a source slice and local index.
    /// Returns `(source, local_index, distance_from_ip)`.
    #[inline(always)]
    fn resolve_candidate<'a>(
        candidate_abs: usize,
        ip_abs: usize,
        input: &'a [u8],
        stream_offset: usize,
        ext_dict: &'a [u8],
        ext_dict_stream_offset: usize,
    ) -> Option<(&'a [u8], usize, usize)> {
        let distance = ip_abs.wrapping_sub(candidate_abs);
        if distance == 0 || distance > MAX_DISTANCE {
            return None;
        }
        if candidate_abs >= stream_offset {
            let local = candidate_abs - stream_offset;
            Some((input, local, distance))
        } else if !ext_dict.is_empty() && candidate_abs >= ext_dict_stream_offset {
            let local = candidate_abs - ext_dict_stream_offset;
            Some((ext_dict, local, distance))
        } else {
            None
        }
    }

    while ip <= mflimit {
        let ip_abs = ip + stream_offset;

        // Try 8-byte hash first
        let h8 = get_hash8_mid(input, ip);
        let pos8_abs = hash8[h8] as usize;
        hash8[h8] = ip_abs as u32;

        if let Some((src8, cand8, dist8)) = resolve_candidate(
            pos8_abs,
            ip_abs,
            input,
            stream_offset,
            ext_dict,
            ext_dict_stream_offset,
        ) {
            let mut probe = ip;
            let match_len = count_same_bytes(input, &mut probe, src8, cand8, match_limit);
            if match_len >= MIN_MATCH {
                let mut cur = ip;
                let mut candidate = cand8;
                let cand_src = src8;
                backtrack_match(input, &mut cur, anchor, cand_src, &mut candidate);
                let match_len = count_same_bytes(input, &mut cur, cand_src, candidate, match_limit);
                let match_start = cur - match_len;
                let offset = dist8 as u16;

                add_hash8(hash8, input, match_start + 1, input_end, stream_offset);
                add_hash8(hash8, input, match_start + 2, input_end, stream_offset);
                add_hash4(hash4, input, match_start + 1, input_end, stream_offset);

                encode_sequence(
                    &input[anchor..match_start],
                    output,
                    offset,
                    match_len - MIN_MATCH,
                );

                ip = cur;
                anchor = ip;

                if ip >= 5 && ip <= ilimit {
                    add_hash8(hash8, input, ip - 5, input_end, stream_offset);
                }
                if ip >= 3 && ip <= ilimit {
                    add_hash8(hash8, input, ip - 3, input_end, stream_offset);
                    add_hash8(hash8, input, ip - 2, input_end, stream_offset);
                }
                if ip >= 2 {
                    add_hash4(hash4, input, ip - 2, input_end, stream_offset);
                }
                if ip >= 1 {
                    add_hash4(hash4, input, ip - 1, input_end, stream_offset);
                }
                continue;
            }
        }

        // Try 4-byte hash
        let h4 = get_hash4_mid(input, ip);
        let pos4_abs = hash4[h4] as usize;
        hash4[h4] = ip_abs as u32;

        if let Some((src4, cand4, dist4)) = resolve_candidate(
            pos4_abs,
            ip_abs,
            input,
            stream_offset,
            ext_dict,
            ext_dict_stream_offset,
        ) {
            let mut probe = ip;
            let match_len = count_same_bytes(input, &mut probe, src4, cand4, match_limit);
            if match_len >= MIN_MATCH {
                let mut best_ip = ip;
                let mut best_src: &[u8] = src4;
                let mut best_cand = cand4;
                let mut best_len = match_len;
                let mut best_dist = dist4;

                if ip + 1 <= mflimit {
                    let h8_next = get_hash8_mid(input, ip + 1);
                    let pos8_next_abs = hash8[h8_next] as usize;
                    if let Some((src8n, cand8n, dist8n)) = resolve_candidate(
                        pos8_next_abs,
                        ip_abs + 1,
                        input,
                        stream_offset,
                        ext_dict,
                        ext_dict_stream_offset,
                    ) {
                        let mut probe_next = ip + 1;
                        let len_next =
                            count_same_bytes(input, &mut probe_next, src8n, cand8n, match_limit);
                        if len_next > best_len {
                            hash8[h8_next] = (ip + 1 + stream_offset) as u32;
                            best_ip = ip + 1;
                            best_src = src8n;
                            best_cand = cand8n;
                            best_len = len_next;
                            best_dist = dist8n;
                        }
                    }
                }
                let _ = best_len;

                let mut cur = best_ip;
                let mut candidate = best_cand;
                backtrack_match(input, &mut cur, anchor, best_src, &mut candidate);
                let match_len = count_same_bytes(input, &mut cur, best_src, candidate, match_limit);
                let match_start = cur - match_len;
                let offset = best_dist as u16;

                add_hash8(hash8, input, match_start + 1, input_end, stream_offset);
                add_hash8(hash8, input, match_start + 2, input_end, stream_offset);
                add_hash4(hash4, input, match_start + 1, input_end, stream_offset);

                encode_sequence(
                    &input[anchor..match_start],
                    output,
                    offset,
                    match_len - MIN_MATCH,
                );

                ip = cur;
                anchor = ip;

                if ip >= 5 && ip <= ilimit {
                    add_hash8(hash8, input, ip - 5, input_end, stream_offset);
                }
                if ip >= 3 && ip <= ilimit {
                    add_hash8(hash8, input, ip - 3, input_end, stream_offset);
                    add_hash8(hash8, input, ip - 2, input_end, stream_offset);
                }
                if ip >= 2 {
                    add_hash4(hash4, input, ip - 2, input_end, stream_offset);
                }
                if ip >= 1 {
                    add_hash4(hash4, input, ip - 1, input_end, stream_offset);
                }
                continue;
            }
        }

        // No match - skip with acceleration
        ip += 1 + ((ip - anchor) >> 9);
    }

    if anchor < input_end {
        handle_last_literals(output, &input[anchor..]);
    }

    Ok(output.pos() - output_start)
}

/// Internal HC compression implementation using hash chain algorithm.
/// `input_pos` is where the current block starts (positions before it are prefix).
/// `ext_dict` and `stream_offset` support linked block mode.
fn compress_hc_internal(
    input: &[u8],
    input_pos: usize,
    output: &mut impl Sink,
    ht: &mut HashTableHCU32,
    ext_dict: &[u8],
    stream_offset: usize,
) -> Result<usize, CompressError> {
    let output_start_pos = output.pos();
    if input.len() - input_pos < MFLIMIT + 1 {
        handle_last_literals(output, &input[input_pos..]);
        return Ok(output.pos() - output_start_pos);
    }

    let src_end = input.len();
    let mf_limit = src_end - MFLIMIT;
    let match_limit = src_end - LAST_LITERALS;

    let mut s_off = input_pos + 1;
    let mut anchor = input_pos;
    let mut match0;
    let mut match1 = Match::new();
    let mut match2 = Match::new();
    let mut match3 = Match::new();

    while s_off < mf_limit {
        if !ht.insert_and_find_best_match(
            input,
            s_off as u32,
            match_limit as u32,
            &mut match1,
            ext_dict,
            stream_offset,
        ) {
            s_off += 1;
            continue;
        }

        match0 = match1;

        loop {
            debug_assert!(match1.start as usize >= anchor);
            if match1.end() > mf_limit
                || !ht.insert_and_find_wider_match(
                    input,
                    (match1.end() - 2) as u32,
                    match1.start,
                    match_limit as u32,
                    match1.len,
                    &mut match2,
                    ext_dict,
                    stream_offset,
                )
            {
                match1.encode_to(&input, anchor, output);
                s_off = match1.end();
                anchor = s_off;
                break;
            }

            if match0.start < match1.start
                && (match2.start as usize) < match1.start as usize + match0.len as usize
            {
                match1 = match0;
            }
            debug_assert!(match2.start >= match1.start);

            if (match2.start - match1.start) < 3 {
                match1 = match2;
                continue;
            }

            let restart = loop {
                if (match2.start - match1.start) < OPTIMAL_ML as u32 {
                    let mut new_match_len = match1.len as usize;
                    if new_match_len > OPTIMAL_ML {
                        new_match_len = OPTIMAL_ML;
                    }
                    if match1.start as usize + new_match_len > match2.end().saturating_sub(MINMATCH)
                    {
                        new_match_len = (match2.start - match1.start) as usize
                            + (match2.len as usize).saturating_sub(MINMATCH);
                    }
                    let correction =
                        new_match_len.saturating_sub((match2.start - match1.start) as usize);
                    if correction > 0 {
                        match2.fix(correction);
                    }
                }

                if match2.end() > mf_limit
                    || !ht.insert_and_find_wider_match(
                        input,
                        (match2.end() - 3) as u32,
                        match2.start,
                        match_limit as u32,
                        match2.len,
                        &mut match3,
                        ext_dict,
                        stream_offset,
                    )
                {
                    if (match2.start as usize) < match1.end() {
                        match1.len = (match2.start - match1.start) as u32;
                    }
                    match1.encode_to(input, anchor, output);
                    s_off = match1.end();
                    anchor = s_off;
                    match2.encode_to(input, anchor, output);
                    s_off = match2.end();
                    anchor = s_off;
                    break false;
                }

                if (match3.start as usize) < match1.end() + 3 {
                    if match3.start as usize >= match1.end() {
                        if (match2.start as usize) < match1.end() {
                            let correction = match1.end() - match2.start as usize;
                            match2.fix(correction);
                            if (match2.len as usize) < MINMATCH {
                                match2 = match3;
                            }
                        }

                        match1.encode_to(input, anchor, output);
                        s_off = match1.end();
                        anchor = s_off;

                        match1 = match3;
                        match0 = match2;

                        break true;
                    }

                    match2 = match3;
                    continue;
                }

                if (match2.start as usize) < match1.end() {
                    if (match2.start - match1.start) < ML_MASK as u32 {
                        if match1.len as usize > OPTIMAL_ML {
                            match1.len = OPTIMAL_ML as u32;
                        }
                        if match1.end() > match2.end() - MINMATCH {
                            match1.len = (match2.end() - match1.start as usize - MINMATCH) as u32;
                        }
                        let correction = match1.end() - match2.start as usize;
                        match2.fix(correction);
                    } else {
                        match1.len = (match2.start - match1.start) as u32;
                    }
                }

                match1.encode_to(input, anchor, output);
                s_off = match1.end();
                anchor = s_off;

                match1 = match2;
                match2 = match3;

                continue;
            };

            if restart {
                continue;
            }
            break;
        }
    }

    handle_last_literals(output, &input[anchor..src_end]);
    Ok(output.pos() - output_start_pos)
}

/// Internal optimal parsing compression implementation
fn compress_opt_internal(
    input: &[u8],
    input_pos: usize,
    output: &mut impl Sink,
    level: u8,
    ht: &mut HashTableHCU32,
    ext_dict: &[u8],
    stream_offset: usize,
) -> Result<usize, CompressError> {
    let output_start_pos = output.pos();

    if input.len() - input_pos < MFLIMIT + 1 {
        handle_last_literals(output, &input[input_pos..]);
        return Ok(output.pos() - output_start_pos);
    }

    let src_end = input.len();
    let mf_limit = src_end - MFLIMIT;
    let match_limit = src_end - LAST_LITERALS;

    // Determine search parameters based on level
    let sufficient_len = match level {
        10 => 64,
        11 => 128,
        _ => LZ4_OPT_NUM, // level 12+
    };

    let full_update = level >= 12;

    let mut anchor = input_pos;
    let mut ip = input_pos;

    let mut opt = vec![OptimalState::SENTINEL; LZ4_OPT_NUM + TRAILING_LITERALS];

    let sufficient_len = sufficient_len.min(LZ4_OPT_NUM - 1);

    'main_loop: while ip <= mf_limit {
        let llen = (ip - anchor) as i32;

        let (first_len, first_off) = ht.find_longer_match(
            input,
            ip as u32,
            match_limit as u32,
            (MIN_MATCH - 1) as u32,
            ext_dict,
            stream_offset,
        );
        if first_len == 0 {
            ip += 1;
            continue;
        }
        let first_len = first_len as usize;

        // If match is good enough, encode immediately
        if first_len >= sufficient_len {
            encode_sequence(&input[anchor..ip], output, first_off, first_len - MIN_MATCH);
            ip += first_len;
            anchor = ip;
            continue;
        }

        // Initialize optimal parsing state for literals
        for rpos in 0..MIN_MATCH as i32 {
            let cost = literals_price(llen + rpos);
            opt[rpos as usize].mlen = 1;
            opt[rpos as usize].off = 0;
            opt[rpos as usize].litlen = llen + rpos;
            opt[rpos as usize].price = cost;
        }

        // Set prices using initial match
        let match_ml = first_len.min(LZ4_OPT_NUM - 1);
        for mlen in MIN_MATCH..=match_ml {
            let cost = sequence_price(llen, mlen as i32);
            opt[mlen].mlen = mlen as i32;
            opt[mlen].off = first_off as i32;
            opt[mlen].litlen = llen;
            opt[mlen].price = cost;
        }

        let mut last_match_pos = first_len;

        // Add trailing literals after the match
        for add_lit in 1..=TRAILING_LITERALS {
            let pos = last_match_pos + add_lit;
            if pos < opt.len() {
                opt[pos].mlen = 1; // literal
                opt[pos].off = 0;
                opt[pos].litlen = add_lit as i32;
                opt[pos].price = opt[last_match_pos].price + literals_price(add_lit as i32);
            }
        }

        // Check further positions
        let mut cur: usize = 1;
        while cur < last_match_pos {
            let cur_ptr = ip + cur;

            if cur_ptr > mf_limit {
                break;
            }

            if full_update {
                // Not useful to search here if next position has same (or lower) cost
                if opt[cur + 1].price <= opt[cur].price
                    && opt[cur + MIN_MATCH].price < opt[cur].price + 3
                {
                    cur += 1;
                    continue;
                }
            } else {
                // Not useful to search here if next position has same (or lower) cost
                if opt[cur + 1].price <= opt[cur].price {
                    cur += 1;
                    continue;
                }
            }

            // Find longer match at current position
            let min_len_search: u32 = if full_update {
                (MIN_MATCH - 1) as u32
            } else {
                (last_match_pos - cur) as u32
            };

            let (new_len, new_off) = ht.find_longer_match(
                input,
                cur_ptr as u32,
                match_limit as u32,
                min_len_search,
                ext_dict,
                stream_offset,
            );
            if new_len == 0 {
                cur += 1;
                continue;
            }
            let new_len = new_len as usize;

            // If match is good enough or extends beyond buffer, encode immediately
            if new_len >= sufficient_len || new_len + cur >= LZ4_OPT_NUM {
                let capped_len = new_len;

                // Set last_match_pos = cur + 1 as in C code
                last_match_pos = cur + 1;

                // Reverse traversal starting from cur
                let mut selected_mlen = capped_len as i32;
                let mut selected_off = new_off as i32;
                let mut candidate_pos = cur;
                loop {
                    let next_mlen = opt[candidate_pos].mlen;
                    let next_off = opt[candidate_pos].off;
                    opt[candidate_pos].mlen = selected_mlen;
                    opt[candidate_pos].off = selected_off;
                    selected_mlen = next_mlen;
                    selected_off = next_off;
                    if (next_mlen as usize) > candidate_pos {
                        break;
                    }
                    candidate_pos -= next_mlen as usize;
                }

                // Encode all recorded sequences in order
                let mut rpos: usize = 0;
                while rpos < last_match_pos {
                    let ml = opt[rpos].mlen as usize;
                    let offset = opt[rpos].off as u16;

                    if ml == 1 {
                        ip += 1;
                        rpos += 1;
                        continue;
                    }

                    encode_sequence(&input[anchor..ip], output, offset, ml - MIN_MATCH);

                    ip += ml;
                    anchor = ip;
                    rpos += ml;
                }

                continue 'main_loop;
            }

            // Update prices for literals before the match
            {
                let base_litlen = opt[cur].litlen;
                for litlen in 1..MIN_MATCH as i32 {
                    let pos = cur + litlen as usize;
                    let price = opt[cur].price - literals_price(base_litlen)
                        + literals_price(base_litlen + litlen);
                    if price < opt[pos].price {
                        opt[pos].mlen = 1; // literal
                        opt[pos].off = 0;
                        opt[pos].litlen = base_litlen + litlen;
                        opt[pos].price = price;
                    }
                }
            }

            // Set prices using match at current position
            {
                let match_ml = new_len.min(LZ4_OPT_NUM - cur - 1);
                for ml in MIN_MATCH..=match_ml {
                    let pos = cur + ml;
                    let (ll, price) = if opt[cur].mlen == 1 {
                        let ll = opt[cur].litlen;
                        let base_price = if cur as i32 > ll {
                            opt[cur - ll as usize].price
                        } else {
                            0
                        };
                        (ll, base_price + sequence_price(ll, ml as i32))
                    } else {
                        (0, opt[cur].price + sequence_price(0, ml as i32))
                    };

                    if pos > last_match_pos + TRAILING_LITERALS || price <= opt[pos].price {
                        if ml == match_ml && last_match_pos < pos {
                            last_match_pos = pos;
                        }
                        opt[pos].mlen = ml as i32;
                        opt[pos].off = new_off as i32;
                        opt[pos].litlen = ll;
                        opt[pos].price = price;
                    }
                }
            }

            // Complete following positions with literals
            for add_lit in 1..=TRAILING_LITERALS as i32 {
                let pos = last_match_pos + add_lit as usize;
                opt[pos].mlen = 1; // literal
                opt[pos].off = 0;
                opt[pos].litlen = add_lit;
                opt[pos].price = opt[last_match_pos].price + literals_price(add_lit);
            }

            cur += 1;
        }

        // Reverse traversal to find the optimal path
        {
            let mut best_mlen = opt[last_match_pos].mlen;
            let mut best_off = opt[last_match_pos].off;
            let mut candidate_pos = last_match_pos - best_mlen as usize;

            loop {
                let next_mlen = opt[candidate_pos].mlen;
                let next_off = opt[candidate_pos].off;
                opt[candidate_pos].mlen = best_mlen;
                opt[candidate_pos].off = best_off;
                best_mlen = next_mlen;
                best_off = next_off;
                if (next_mlen as usize) > candidate_pos {
                    break;
                }
                candidate_pos -= next_mlen as usize;
            }
        }

        // Encode all recorded sequences in order
        {
            let mut rpos: usize = 0;
            while rpos < last_match_pos {
                let ml = opt[rpos].mlen as usize;
                let offset = opt[rpos].off as u16;

                if ml == 1 {
                    ip += 1;
                    rpos += 1;
                    continue;
                }

                encode_sequence(&input[anchor..ip], output, offset, ml - MIN_MATCH);

                ip += ml;
                anchor = ip;
                rpos += ml;
            }
        }

        // No opt array reset needed (matches C behavior)
    }

    // Handle remaining literals
    handle_last_literals(output, &input[anchor..src_end]);
    Ok(output.pos() - output_start_pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sink::SliceSink;

    #[test]
    fn test_compress_hc_basic() {
        let input = b"Hello, this is a test string that should be compressed!";
        let mut output = vec![0u8; input.len() * 2]; // Ensure enough space
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 17);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_small_input() {
        let input = b"Hi"; // Too small to compress
        let mut output = vec![0u8; 100];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 17);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_repeated_pattern() {
        let input = b"AAAAAAAAAAABBBBBAAABBBBBBBAAAAAAA"; // Highly compressible
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 17);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len() * 8);
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_10() {
        // Level 10 uses optimal parsing
        let input = b"Hello, this is a test string that should be compressed!";
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 10);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_10_small_input() {
        let input = b"Hi"; // Too small to compress
        let mut output = vec![0u8; 100];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 10);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_10_repeated_pattern() {
        let input = b"AAAAAAAAAAABBBBBAAABBBBBBBAAAAAAA"; // Highly compressible
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 10);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len() * 8);
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_11() {
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 11);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_12() {
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(input, &mut sink, 12);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_12_better_than_level_9() {
        // Level 12 (optimal) should produce same or smaller output than level 9 (HC)
        let input: Vec<u8> = (0..1000)
            .map(|i| {
                let patterns = [b"ABCD", b"EFGH", b"IJKL", b"MNOP"];
                patterns[(i / 50) % 4][i % 4]
            })
            .collect();

        let mut output_hc = vec![0u8; input.len() * 2];
        let mut sink_hc = SliceSink::new(&mut output_hc, 0);
        let hc_size = compress_hc(&input, &mut sink_hc, 9).unwrap();

        let mut output_opt = vec![0u8; input.len() * 2];
        let mut sink_opt = SliceSink::new(&mut output_opt, 0);
        let opt_size = compress_hc(&input, &mut sink_opt, 12).unwrap();

        // Optimal should produce same or smaller output
        assert!(
            opt_size <= hc_size,
            "Level 12 ({}) should be <= Level 9 ({})",
            opt_size,
            hc_size
        );

        // Both should decompress correctly
        let result_hc = decompress(&output_hc[..hc_size], input.len());
        assert!(result_hc.is_ok());
        assert_eq!(&input[..], &result_hc.unwrap()[..]);

        let result_opt = decompress(&output_opt[..opt_size], input.len());
        assert!(result_opt.is_ok());
        assert_eq!(&input[..], &result_opt.unwrap()[..]);
    }

    #[test]
    fn test_compress_hc_level_10_large_input() {
        // Test with a larger input to exercise the optimal algorithm
        let input: Vec<u8> = (0..10000).map(|i| ((i * 7 + 13) % 256) as u8).collect();

        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(&input, &mut sink, 10);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_10_all_same() {
        // Test with all same bytes - highly compressible
        let input = vec![0x42u8; 5000];
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);

        let result = compress_hc(&input, &mut sink, 10);
        assert!(result.is_ok());

        let compressed_size = result.unwrap();
        assert!(compressed_size > 0);
        // Should compress very well
        assert!(
            compressed_size < input.len() / 10,
            "Should compress very well"
        );

        let result = decompress(&output[..compressed_size], input.len());
        assert!(result.is_ok());
        assert_eq!(&input[..], &result.unwrap()[..])
    }

    #[test]
    fn test_compress_hc_level_clamping() {
        // Test that levels are clamped correctly
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";

        // Level 0 should be clamped to 1
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);
        let result = compress_hc(input, &mut sink, 0);
        assert!(result.is_ok());
        let size_level_0 = result.unwrap();
        let decompressed = decompress(&output[..size_level_0], input.len()).unwrap();
        assert_eq!(&input[..], &decompressed[..]);

        // Level 20 should be clamped to 12
        let mut output = vec![0u8; input.len() * 2];
        let mut sink = SliceSink::new(&mut output, 0);
        let result = compress_hc(input, &mut sink, 20);
        assert!(result.is_ok());
        let size_level_20 = result.unwrap();
        let decompressed = decompress(&output[..size_level_20], input.len()).unwrap();
        assert_eq!(&input[..], &decompressed[..]);
    }
}

#[cfg(test)]
#[test]
fn test_lz4mid_debug() {
    use crate::sink::SliceSink;
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
    println!("Input len: {}", input.len());
    println!("Input: {:?}", String::from_utf8_lossy(input));

    let mut output = vec![0u8; input.len() * 2];
    let mut sink = SliceSink::new(&mut output, 0);
    let mut table = HashTableMid::new();
    let size = compress_mid_internal(input, 0, &mut sink, &mut table, &[], 0).unwrap();
    println!("Compressed size: {}", size);
    println!("Compressed: {:02x?}", &output[..size]);

    // Try to decompress
    match decompress(&output[..size], input.len()) {
        Ok(d) => {
            println!("Decompressed: {} bytes", d.len());
            println!("Match: {}", d == input);
        }
        Err(e) => println!("Decompress error: {:?}", e),
    }
}
