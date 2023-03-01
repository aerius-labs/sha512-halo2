use super::super::AssignedBits;
use super::MessageScheduleConfig;
use halo2_proofs::{
    circuit::{Region, Value},
    halo2curves::bn256,
    plonk::Error,
};

#[cfg(test)]
use super::super::{super::BLOCK_SIZE, BlockWord, ROUNDS};

// Rows needed for each gate
pub const DECOMPOSE_0_ROWS: usize = 6;
pub const DECOMPOSE_1_ROWS: usize = 6;
pub const DECOMPOSE_2_ROWS: usize = 6;
pub const DECOMPOSE_3_ROWS: usize = 6;
pub const SIGMA_0_V1_ROWS: usize = 8;
pub const SIGMA_0_V2_ROWS: usize = 8;
pub const SIGMA_1_V1_ROWS: usize = 8;
pub const SIGMA_1_V2_ROWS: usize = 8;

// Rows needed for each subregion
pub const SUBREGION_0_LEN: usize = 1; // W_0
pub const SUBREGION_0_ROWS: usize = SUBREGION_0_LEN * DECOMPOSE_0_ROWS;
pub const SUBREGION_1_WORD: usize = DECOMPOSE_1_ROWS + SIGMA_0_V1_ROWS;
pub const SUBREGION_1_LEN: usize = 13; // W_[1..14]
pub const SUBREGION_1_ROWS: usize = SUBREGION_1_LEN * SUBREGION_1_WORD;
pub const SUBREGION_2_WORD: usize = DECOMPOSE_2_ROWS + SIGMA_0_V2_ROWS + SIGMA_1_V2_ROWS;
pub const SUBREGION_2_LEN: usize = 51; // W_[14..65]
pub const SUBREGION_2_ROWS: usize = SUBREGION_2_LEN * SUBREGION_2_WORD;
pub const SUBREGION_3_WORD: usize = DECOMPOSE_3_ROWS + SIGMA_1_V1_ROWS;
pub const SUBREGION_3_LEN: usize = 13; // W[65..78]
pub const SUBREGION_3_ROWS: usize = SUBREGION_3_LEN * SUBREGION_3_WORD;
// pub const SUBREGION_4_LEN: usize = 2; // W_[78..80]
// pub const SUBREGION_4_ROWS: usize = SUBREGION_4_LEN * DECOMPOSE_0_ROWS;

/// Returns row number of a word
pub fn get_word_row(word_idx: usize) -> usize {
    assert!(word_idx <= 79);
    if word_idx == 0 {
        0
    } else if (1..=13).contains(&word_idx) {
        SUBREGION_0_ROWS + SUBREGION_1_WORD * (word_idx - 1)
    } else if (14..=64).contains(&word_idx) {
        SUBREGION_0_ROWS + SUBREGION_1_ROWS + SUBREGION_2_WORD * (word_idx - 14) + 1
    } else if (65..=77).contains(&word_idx) {
        SUBREGION_0_ROWS + SUBREGION_1_ROWS + SUBREGION_2_ROWS + SUBREGION_3_WORD * (word_idx - 65)
    } else {
        SUBREGION_0_ROWS
            + SUBREGION_1_ROWS
            + SUBREGION_2_ROWS
            + SUBREGION_3_ROWS
            + DECOMPOSE_0_ROWS * (word_idx - 78)
    }
}

/// Test vector: "abc"
#[cfg(test)]
pub fn msg_schedule_test_input() -> [BlockWord; BLOCK_SIZE] {
    [
        BlockWord(Value::known(0b0110000101100010011000111000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000000000)),
        BlockWord(Value::known(0b0000000000000000000000000000000000000000000000000000000000011000)),
    ]
}


#[cfg(test)]
pub const MSG_SCHEDULE_TEST_OUTPUT: [u64; ROUNDS] = [
    0b0110000101100010011000111000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000000000,
    0b0000000000000000000000000000000000000000000000000000000000011000,
    0b0110000101100010011000111000000000000000000000000000000000000000,
    0b0000000000000011000000000000000000000000000000000000000011000000,
    0b0000101010010110100110011010001001001100011100000000000000000011,
    0b0000000000000000000011000000000001100000000000000000011000000011,
    0b0101010010011110111101100010011000111001100001011000100110010110,
    0b0000000011000000000000000011001100000000000000000011110000000000,
    0b0001010010010111000000000111101010001010000011101001110110111100,
    0b0110001011100101011001010000000011001100000001111000000011110000,
    0b0111011101100000110111010100011101011010010100111000011110010111,
    0b1111000101010101010010110111000100011100000111000000000000000011,
    0b1100101000101001100100111010010000110100010111011001111111110010,
    0b0101111000001110011001101011010111000111100000111101110100110010,
    0b1110001001011010011000100101110100000000010010010100101101100010,
    0b1001111101000100010010000110111110110001111001001111101111010010,
    0b1011001100011011100011000010101100000110000010000101111100101111,
    0b0000111010011000011101100110000010010011010000010100001011110110,
    0b1010010010101111001011001111110100001001111110111011100100100100,
    0b1010110100101000100111100010111000001011110101010011000110000110,
    0b0011110001110100010101100011101010100010111110010110011100111110,
    0b0110110011001101110011010001010011001100000101001011010100111111,
    0b1100001111111001001001011011001100110111111100100010101111011110,
    0b0101101111001100011101111010011101011010110110010101101101010100,
    0b0011111011000010001001010111101011011100101000001001101001010010,
    0b0010100000100100011010010110000000000000000111111100010111101011,
    0b0000010011100011001110100111010111001110001010111110100010001010,
    0b0111110101010011000101001011001111000011010110011110000011100111,
    0b1010111011110111101000101000010111111111001001010001001001100110,
    0b0000101110000100011100100101100000011101111011101010000001001111,
    0b1011000101110100111000100110111011011101110001111011000000110011,
    0b0101110101100011101110101110010110001101110111011000100011011110,
    0b0100110000000100010000000000011110110111010001001100110010111011,
    0b1110011010101001101010100100110101110100110111000111110101000011,
    0b1110101111101010111100010010001101110010010010000000000110011100,
    0b0011011000011110100000001011001011010000000011110011000110010011,
    0b0010111010011000001110010001001001011101111100111011000101110101,
    0b0011001100011001011000101001001010010011101011010101001101100011,
    0b1001110010111100010111011000100110101100000110111000100111010101,
    0b0010011101011110001000111111111111101110110010100101000010110111,
    0b0011101110000000110101101000000010111111011010011110111101011000,
    0b0000110100000110100101101001001100111001010001011010000100100101,
    0b0111010100110011111010101011110010110111100001101111111100000000,
    0b1011100010011000001001101100111011100110111110111111000011100101,
    0b0010010010011011010011111011110010101101011000100011111010011111,
    0b0100101011101010100111011111001010110000001011010110111100011110,
    0b0010110011000101011101000111010110100101010111101000110110001111,
    0b1011001001010111010010101110100100111000110110001011111010001001,
    0b1100000110110011010110100101011110110001011011010110101011101010,
    0b1100110001001001000110001011010110010100100100100000011010111011,
    0b0101000010011001110000111010110111010111100111111001000011101100,
    0b0101111010101000000111010111100011100111011001100000101111110001,
    0b1110101111101110011000100110011101000000010110101100001010101001,
    0b1011000000011111001000011001001001100001000010001010010010101011,
    0b0111100001100100001100111101110100101111111001100101010101010110,
    0b1100010101001010011011101010101000100100101000000101010100101100,
    0b1011001111001000111100010101001100001011110110111010101010011110,
    0b1011101110001010101111111110010101101111010001101001001100111000,
    0b1111011000111101010000100110010111001100000111000101101001111000,
    0b1011111010000011010101011110101001110011000100101001101011111011,
    0b0100100111100010110110111000111010111101110011111011111010110101,
    0b1000001000100110100111010100101010001000001110100011110110011001,
    0b1111110111110101001111011111001100000001000111110011011000101011,
    0b0100011001001010111101010110011100011101011100011100000100101110,
    0b1110010001001001101101101000000110011000111011000110000100011100,
    0b1001001010101110111011101101000110100111101111001111011111010010,
];

impl MessageScheduleConfig {
    // Assign a word and its hi and lo halves
    pub fn assign_word_and_halves(
        &self,
        region: &mut Region<'_, bn256::Fr>,
        word: Value<u64>,
        word_idx: usize,
    ) -> Result<(AssignedBits<64>, (AssignedBits<32>, AssignedBits<32>)), Error> {
        // Rename these here for ease of matching the gates to the specification.
        let a_3 = self.extras[0];
        let a_4 = self.extras[1];

        let row = get_word_row(word_idx);

        let w_lo = {
            let w_lo_val = word.map(|word| word as u32);
            AssignedBits::<32>::assign(region, || format!("W_{}_lo", word_idx), a_3, row, w_lo_val)?
        };
        let w_hi = {
            let w_hi_val = word.map(|word| (word >> 32) as u32);
            AssignedBits::<32>::assign(region, || format!("W_{}_hi", word_idx), a_4, row, w_hi_val)?
        };

        let word = AssignedBits::<64>::assign(
            region,
            || format!("W_{}", word_idx),
            self.message_schedule,
            row,
            word,
        )?;

        Ok((word, (w_lo, w_hi)))
    }
}
