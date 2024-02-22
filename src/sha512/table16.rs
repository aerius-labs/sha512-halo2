use std::convert::TryInto;
use std::marker::PhantomData;
use super::Sha512Instructions;
use halo2_proofs::{
    circuit::{AssignedCell, Chip, Layouter, Region, Value},
    halo2curves::bn256,
    plonk::{Advice, Any, Assigned, Column, ConstraintSystem, Error},
};

pub mod compression;
use compression::{compression_util::match_state, CompressionConfig};
mod gates;
mod message_schedule;
mod spread_table;
mod util;

pub use compression::*;
use gates::*;
use message_schedule::*;
use spread_table::*;
use util::*;

const ROUNDS: usize = 80;
const STATE: usize = 8;

#[allow(clippy::unreadable_literal)]
pub(crate) const ROUND_CONSTANTS: [u64; ROUNDS] = [
    0x428a2f98d728ae22,
    0x7137449123ef65cd,
    0xb5c0fbcfec4d3b2f,
    0xe9b5dba58189dbbc,
    0x3956c25bf348b538,
    0x59f111f1b605d019,
    0x923f82a4af194f9b,
    0xab1c5ed5da6d8118,
    0xd807aa98a3030242,
    0x12835b0145706fbe,
    0x243185be4ee4b28c,
    0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f,
    0x80deb1fe3b1696b1,
    0x9bdc06a725c71235,
    0xc19bf174cf692694,
    0xe49b69c19ef14ad2,
    0xefbe4786384f25e3,
    0x0fc19dc68b8cd5b5,
    0x240ca1cc77ac9c65,
    0x2de92c6f592b0275,
    0x4a7484aa6ea6e483,
    0x5cb0a9dcbd41fbd4,
    0x76f988da831153b5,
    0x983e5152ee66dfab,
    0xa831c66d2db43210,
    0xb00327c898fb213f,
    0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2,
    0xd5a79147930aa725,
    0x06ca6351e003826f,
    0x142929670a0e6e70,
    0x27b70a8546d22ffc,
    0x2e1b21385c26c926,
    0x4d2c6dfc5ac42aed,
    0x53380d139d95b3df,
    0x650a73548baf63de,
    0x766a0abb3c77b2a8,
    0x81c2c92e47edaee6,
    0x92722c851482353b,
    0xa2bfe8a14cf10364,
    0xa81a664bbc423001,
    0xc24b8b70d0f89791,
    0xc76c51a30654be30,
    0xd192e819d6ef5218,
    0xd69906245565a910,
    0xf40e35855771202a,
    0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8,
    0x1e376c085141ab53,
    0x2748774cdf8eeb99,
    0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63,
    0x4ed8aa4ae3418acb,
    0x5b9cca4f7763e373,
    0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc,
    0x78a5636f43172f60,
    0x84c87814a1f0ab72,
    0x8cc702081a6439ec,
    0x90befffa23631e28,
    0xa4506cebde82bde9,
    0xbef9a3f7b2c67915,
    0xc67178f2e372532b,
    0xca273eceea26619c,
    0xd186b8c721c0c207,
    0xeada7dd6cde0eb1e,
    0xf57d4f7fee6ed178,
    0x06f067aa72176fba,
    0x0a637dc5a2c898a6,
    0x113f9804bef90dae,
    0x1b710b35131c471b,
    0x28db77f523047d84,
    0x32caab7b40c72493,
    0x3c9ebe0a15c9bebc,
    0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6,
    0x597f299cfc657e2a,
    0x5fcb6fab3ad6faec,
    0x6c44198c4a475817,
];

pub const IV: [u64; STATE] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

#[derive(Clone, Copy, Debug, Default)]
/// A word in a `Table16` message block.
// TODO: Make the internals of this struct private.
pub struct BlockWord(pub Value<u64>);

#[derive(Clone, Debug)]
/// Little-endian bits (up to 64 bits)
pub struct Bits<const LEN: usize>([bool; LEN]);

impl<const LEN: usize> Bits<LEN> {
    fn spread<const SPREAD: usize>(&self) -> [bool; SPREAD] {
        spread_bits(self.0)
    }
}

impl<const LEN: usize> std::ops::Deref for Bits<LEN> {
    type Target = [bool; LEN];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const LEN: usize> From<[bool; LEN]> for Bits<LEN> {
    fn from(bits: [bool; LEN]) -> Self {
        Self(bits)
    }
}

impl<const LEN: usize> From<&Bits<LEN>> for [bool; LEN] {
    fn from(bits: &Bits<LEN>) -> Self {
        bits.0
    }
}

impl<const LEN: usize> From<&Bits<LEN>> for Assigned<bn256::Fr> {
    fn from(bits: &Bits<LEN>) -> Assigned<bn256::Fr> {
        assert!(LEN <= 64);
        bn256::Fr::from(lebs2ip(&bits.0) as u64).into()
    }
}
impl From<&Bits<16>> for u16 {
    fn from(bits: &Bits<16>) -> u16 {
        lebs2ip(&bits.0) as u16
    }
}
impl From<u16> for Bits<16> {
    fn from(int: u16) -> Bits<16> {
        Bits(i2lebsp::<16>(int.into()))
    }
}
impl From<&Bits<32>> for u32 {
    fn from(bits: &Bits<32>) -> u32 {
        lebs2ip(&bits.0) as u32
    }
}
impl From<u32> for Bits<32> {
    fn from(int: u32) -> Bits<32> {
        Bits(i2lebsp::<32>(int.into()))
    }
}
impl From<&Bits<64>> for u64 {
    fn from(bits: &Bits<64>) -> u64 {
        lebs2ip(&bits.0) as u64
    }
}
impl From<u64> for Bits<64> {
    fn from(int: u64) -> Bits<64> {
        Bits(i2lebsp::<64>(int.into()))
    }
}

#[derive(Clone, Debug)]
pub struct AssignedBits<const LEN: usize>(AssignedCell<Bits<LEN>, bn256::Fr>);

impl<const LEN: usize> std::ops::Deref for AssignedBits<LEN> {
    type Target = AssignedCell<Bits<LEN>, bn256::Fr>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const LEN: usize> AssignedBits<LEN> {
    fn assign_bits<A, AR, T: TryInto<[bool; LEN]> + std::fmt::Debug + Clone>(
        region: &mut Region<'_, bn256::Fr>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<T>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
        <T as TryInto<[bool; LEN]>>::Error: std::fmt::Debug,
    {
        let value: Value<[bool; LEN]> = value.map(|v| v.try_into().unwrap());
        let value: Value<Bits<LEN>> = value.map(|v| v.into());

        let column: Column<Any> = column.into();
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}
#[allow(dead_code)]
impl AssignedBits<16> {
    fn value_u16(&self) -> Value<u16> {
        self.value().map(|v| v.into())
    }
    fn assign<A, AR>(
        region: &mut Region<'_, bn256::Fr>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<u16>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
    {
        let column: Column<Any> = column.into();
        let value: Value<Bits<16>> = value.map(|v| v.into());
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}
impl AssignedBits<32> {
    fn value_u32(&self) -> Value<u32> {
        self.value().map(|v| v.into())
    }
    fn assign<A, AR>(
        region: &mut Region<'_, bn256::Fr>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<u32>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
    {
        let column: Column<Any> = column.into();
        let value: Value<Bits<32>> = value.map(|v| v.into());
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}
impl AssignedBits<64> {
    fn value_u64(&self) -> Value<u64> {
        self.value().map(|v| v.into())
    }
    fn assign<A, AR>(
        region: &mut Region<'_, bn256::Fr>,
        annotation: A,
        column: impl Into<Column<Any>>,
        offset: usize,
        value: Value<u64>,
    ) -> Result<Self, Error>
    where
        A: Fn() -> AR,
        AR: Into<String>,
    {
        let column: Column<Any> = column.into();
        let value: Value<Bits<64>> = value.map(|v| v.into());
        match column.column_type() {
            Any::Advice(_) => {
                region.assign_advice(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            Any::Fixed => {
                region.assign_fixed(annotation, column.try_into().unwrap(), offset, || {
                    value.clone()
                })
            }
            _ => panic!("Cannot assign to instance column"),
        }
        .map(AssignedBits)
    }
}

/// Configuration for a [`Table16Chip`].
#[derive(Clone, Debug)]
pub struct Table16Config {
    lookup: SpreadTableConfig,
    message_schedule: MessageScheduleConfig,
    compression: CompressionConfig,
}
/// A chip that implements SHA-512 with a maximum lookup table size of $2^16$.
#[derive(Clone, Debug)]
pub struct Table16Chip {
    config: Table16Config,
    _marker: PhantomData<bn256::Fr>,
}

impl Chip<bn256::Fr> for Table16Chip {
    type Config = Table16Config;
    type Loaded = ();

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn loaded(&self) -> &Self::Loaded {
        &()
    }
}

impl Table16Chip {
    /// Reconstructs this chip from the given config.
    pub fn construct(config: <Self as Chip<bn256::Fr>>::Config) -> Self {
        Self {
            config,
            _marker: PhantomData,
        }
    }

    /// Configures a circuit to include this chip.
    pub fn configure(meta: &mut ConstraintSystem<bn256::Fr>) -> <Self as Chip<bn256::Fr>>::Config {
        // Columns required by this chip:
        let message_schedule = meta.advice_column();
        let extras = [
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
            meta.advice_column(),
        ];

        // - Three advice columns to interact with the lookup table.
        let input_tag = meta.advice_column();
        let input_dense = meta.advice_column();
        let input_spread = meta.advice_column();

        let lookup = SpreadTableChip::configure(meta, input_tag, input_dense, input_spread);
        let lookup_inputs = lookup.input.clone();

        // Rename these here for ease of matching the gates to the specification.
        let _a_0 = lookup_inputs.tag;
        let a_1 = lookup_inputs.dense;
        let a_2 = lookup_inputs.spread;
        let a_3 = extras[0];
        let a_4 = extras[1];
        let a_5 = message_schedule;
        let a_6 = extras[2];
        let a_7 = extras[3];
        let a_8 = extras[4];
        let _a_9 = extras[5];

        // Add all advice columns to permutation
        for column in [a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8].iter() {
            meta.enable_equality(*column);
        }

        let compression =
            CompressionConfig::configure(meta, lookup_inputs.clone(), message_schedule, extras);

        let message_schedule =
            MessageScheduleConfig::configure(meta, lookup_inputs, message_schedule, extras);

        Table16Config {
            lookup,
            message_schedule,
            compression,
        }
    }

    /// Loads the lookup table required by this chip into the circuit.
    pub fn load(
        config: Table16Config,
        layouter: &mut impl Layouter<bn256::Fr>,
    ) -> Result<(), Error> {
        SpreadTableChip::load(config.lookup, layouter)
    }
}

impl Sha512Instructions<bn256::Fr> for Table16Chip {
    type State = State;
    type BlockWord = BlockWord;

    fn initialization_vector(
        &self,
        layouter: &mut impl Layouter<bn256::Fr>,
    ) -> Result<State, Error> {
        println!("RUNNING initialization_vector FUNCTION");
        self.config().compression.initialize_with_iv(layouter, IV)
    }

    fn initialization(
        &self,
        layouter: &mut impl Layouter<bn256::Fr>,
        init_state: &Self::State,
    ) -> Result<Self::State, Error> {
        println!("RUNNING INITIALIZATION");               
        self.config()
            .compression
            .initialize_with_state(layouter, init_state.clone())
    }

    // Given an initialized state and an input message block, compress the
    // message block and return the final state.
    fn compress(
        &self,
        layouter: &mut impl Layouter<bn256::Fr>,
        initialized_state: &Self::State,
        input: [Self::BlockWord; super::BLOCK_SIZE],
    ) -> Result<Self::State, Error> {
        let config = self.config();
        let (_, w_halves) = config.message_schedule.process(layouter, input)?;
        config
            .compression
            .compress(layouter, initialized_state.clone(), w_halves)
    }

    fn digest(
        &self,
        layouter: &mut impl Layouter<bn256::Fr>,
        state: &Self::State,
    ) -> Result<[Self::BlockWord; super::DIGEST_SIZE], Error> {
        // Copy the dense forms of the state variable chunks down to this gate.
        // Reconstruct the 64-bit dense words.
        self.config().compression.digest(layouter, state.clone())
    }
}

/// Common assignment patterns used by Table16 regions.
trait Table16Assignment {
    fn joindense(x: &SpreadVar<16, 32>, y: &SpreadVar<16, 32>) -> Value<[bool; 32]> {
        x.dense.value().zip(y.dense.value()).map(|(x, y)| {
            x.iter()
                .chain(y.iter())
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
    }

    fn joinspread(x: &SpreadVar<16, 32>, y: &SpreadVar<16, 32>) -> Value<[bool; 64]> {
        x.spread.value().zip(y.spread.value()).map(|(x, y)| {
            x.iter()
                .chain(y.iter())
                .copied()
                .collect::<Vec<_>>()
                .try_into()
                .unwrap()
        })
    }
    /// Assign cells for general spread computation used in sigma, ch, ch_neg, maj gates
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn assign_spread_outputs(
        &self,
        region: &mut Region<'_, bn256::Fr>,
        lookup: &SpreadInputs,
        a_3: Column<Advice>,
        row: usize,
        r_0_even: Value<[bool; 32]>,
        r_0_odd: Value<[bool; 32]>,
        r_1_even: Value<[bool; 32]>,
        r_1_odd: Value<[bool; 32]>,
    ) -> Result<
        (
            (AssignedBits<32>, AssignedBits<32>),
            (AssignedBits<32>, AssignedBits<32>),
        ),
        Error,
    > {
        // Lookup R_0^{even}, R_0^{odd}, R_1^{even}, R_1^{odd}
        let r_0_even_lo: Value<[bool; 16]> =
            r_0_even.map(|r_0_even| r_0_even[..16].try_into().unwrap());
        let r_0_even_hi: Value<[bool; 16]> =
            r_0_even.map(|r_0_even| r_0_even[16..32].try_into().unwrap());

        let r_0_even_lo = SpreadVar::with_lookup(
            region,
            lookup,
            row - 1,
            r_0_even_lo.map(SpreadWord::<16, 32>::new),
        )?;
        let r_0_even_hi = SpreadVar::with_lookup(
            region,
            lookup,
            row,
            r_0_even_hi.map(SpreadWord::<16, 32>::new),
        )?;

        let r_0_even_dense = Self::joindense(&r_0_even_lo, &r_0_even_hi);
        let r_0_even_d =
            AssignedBits::<32>::assign_bits(region, || "r_0_even_d", a_3, row + 2, r_0_even_dense)?;

        let r_0_odd_lo: Value<[bool; 16]> =
            r_0_odd.map(|r_0_odd| r_0_odd[..16].try_into().unwrap());
        let r_0_odd_hi: Value<[bool; 16]> =
            r_0_odd.map(|r_0_odd| r_0_odd[16..32].try_into().unwrap());

        let r_0_odd_lo = SpreadVar::with_lookup(
            region,
            lookup,
            row + 1,
            r_0_odd_lo.map(SpreadWord::<16, 32>::new),
        )?;
        let r_0_odd_hi = SpreadVar::with_lookup(
            region,
            lookup,
            row + 2,
            r_0_odd_hi.map(SpreadWord::<16, 32>::new),
        )?;

        let r_0_odd_dense = Self::joindense(&r_0_odd_lo, &r_0_odd_hi);
        let r_0_odd_d =
            AssignedBits::<32>::assign_bits(region, || "r_0_odd_d", a_3, row + 3, r_0_odd_dense)?;

        let r_1_even_lo: Value<[bool; 16]> =
            r_1_even.map(|r_1_even| r_1_even[..16].try_into().unwrap());
        let r_1_even_hi: Value<[bool; 16]> =
            r_1_even.map(|r_1_even| r_1_even[16..32].try_into().unwrap());

        let r_1_even_lo = SpreadVar::with_lookup(
            region,
            lookup,
            row + 3,
            r_1_even_lo.map(SpreadWord::<16, 32>::new),
        )?;
        let r_1_even_hi = SpreadVar::with_lookup(
            region,
            lookup,
            row + 4,
            r_1_even_hi.map(SpreadWord::<16, 32>::new),
        )?;

        let r_1_even_dense = Self::joindense(&r_1_even_lo, &r_1_even_hi);
        let r_1_even_d =
            AssignedBits::<32>::assign_bits(region, || "r_1_even_d", a_3, row + 4, r_1_even_dense)?;

        let r_1_odd_lo: Value<[bool; 16]> =
            r_1_odd.map(|r_1_odd| r_1_odd[..16].try_into().unwrap());
        let r_1_odd_hi: Value<[bool; 16]> =
            r_1_odd.map(|r_1_odd| r_1_odd[16..32].try_into().unwrap());

        let r_1_odd_lo = SpreadVar::with_lookup(
            region,
            lookup,
            row + 5,
            r_1_odd_lo.map(SpreadWord::<16, 32>::new),
        )?;
        let r_1_odd_hi = SpreadVar::with_lookup(
            region,
            lookup,
            row + 6,
            r_1_odd_hi.map(SpreadWord::<16, 32>::new),
        )?;

        let r_1_odd_dense = Self::joindense(&r_1_odd_lo, &r_1_odd_hi);
        let r_1_odd_d =
            AssignedBits::<32>::assign_bits(region, || "r_1_odd_d", a_3, row + 5, r_1_odd_dense)?;

        let r_1_odd_spread = Self::joinspread(&r_1_odd_lo, &r_1_odd_hi);
        let _r_1_odd_s =
            AssignedBits::<64>::assign_bits(region, || "r_1_odd_d", a_3, row + 6, r_1_odd_spread)?;

        Ok(((r_0_even_d, r_1_even_d), (r_0_odd_d, r_1_odd_d)))
    }

    /// Assign outputs of sigma gates
    #[allow(clippy::too_many_arguments)]
    fn assign_sigma_outputs(
        &self,
        region: &mut Region<'_, bn256::Fr>,
        lookup: &SpreadInputs,
        a_3: Column<Advice>,
        row: usize,
        r_0_even: Value<[bool; 32]>,
        r_0_odd: Value<[bool; 32]>,
        r_1_even: Value<[bool; 32]>,
        r_1_odd: Value<[bool; 32]>,
    ) -> Result<(AssignedBits<32>, AssignedBits<32>), Error> {
        let (even, _odd) = self.assign_spread_outputs(
            region, lookup, a_3, row, r_0_even, r_0_odd, r_1_even, r_1_odd,
        )?;

        Ok(even)
    }
}

#[cfg(test)]
#[cfg(feature = "test-dev-graph")]
mod tests {
    use super::super::{Sha512, BLOCK_SIZE};
    use super::{message_schedule::msg_schedule_test_input, Table16Chip, Table16Config};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        halo2curves::bn256,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    #[test]
    fn print_sha512_circuit() {
        use plotters::prelude::*;
        struct MyCircuit {}

        impl Circuit<bn256::Fr> for MyCircuit {
            type Config = Table16Config;
            type FloorPlanner = SimpleFloorPlanner;

            fn without_witnesses(&self) -> Self {
                MyCircuit {}
            }

            fn configure(meta: &mut ConstraintSystem<bn256::Fr>) -> Self::Config {
                Table16Chip::configure(meta)
            }

            fn synthesize(
                &self,
                config: Self::Config,
                mut layouter: impl Layouter<bn256::Fr>,
            ) -> Result<(), Error> {
                let table16_chip = Table16Chip::construct(config.clone());
                Table16Chip::load(config, &mut layouter)?;

                // Test vector: "abc"
                let test_input = msg_schedule_test_input();

                // Create a message of length 63 blocks
                let mut input = Vec::with_capacity(63 * BLOCK_SIZE);
                for _ in 0..63 {
                    input.extend_from_slice(&test_input);
                }

                Sha512::digest(table16_chip, layouter.namespace(|| "'abc' * 63"), &input)?;

                Ok(())
            }
        }
        let root =
            BitMapBackend::new("sha-512-table16-chip-layout.png", (1024, 3480)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root
            .titled("16-bit Table SHA-512 Chip", ("sans-serif", 60))
            .unwrap();

        let circuit = MyCircuit {};
        halo2_proofs::dev::CircuitLayout::default()
            .render::<bn256::Fr, _, _>(19, &circuit, &root)
            .unwrap();
    }
}
