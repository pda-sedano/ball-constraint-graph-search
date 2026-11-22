//! The sampler's random numbers: xoshiro256**, seeded by SplitMix64.
//!
//! Written out here rather than taken from a crate so that a seed gives the
//! same run forever, whatever happens to dependency versions. Both generators
//! are public domain (Blackman and Vigna, <https://prng.di.unimi.it/>).

/// xoshiro256**: fast, 256 bits of state, and statistically sound for
/// simulation (not for cryptography).
#[derive(Debug, Clone)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// A generator whose whole state is derived from `seed` by SplitMix64, as
    /// the xoshiro authors recommend, so nearby seeds give unrelated streams.
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Rng {
            state: [next(), next(), next(), next()],
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.state;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform on [0, 1), with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on `0..n` by Lemire's multiply-shift. The bias is at most
    /// n / 2^64, far below anything a run could detect. Panics if n is 0.
    pub fn below(&mut self, n: usize) -> usize {
        assert!(n > 0, "below(0)");
        ((u128::from(self.next_u64()) * n as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_output() {
        //! xoshiro256** from state [1, 2, 3, 4], as in the reference C code.
        let mut rng = Rng {
            state: [1, 2, 3, 4],
        };
        let first: Vec<u64> = (0..4).map(|_| rng.next_u64()).collect();
        assert_eq!(first, [11520, 0, 1509978240, 1215971899390074240]);
    }

    #[test]
    fn same_seed_same_stream() {
        let (mut a, mut b, mut c) = (Rng::new(7), Rng::new(7), Rng::new(8));
        let xs: Vec<u64> = (0..5).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..5).map(|_| b.next_u64()).collect();
        let zs: Vec<u64> = (0..5).map(|_| c.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
    }

    #[test]
    fn below_and_next_f64_are_roughly_uniform() {
        let mut rng = Rng::new(1);
        let mut counts = [0usize; 6];
        let draws = 600_000;
        for _ in 0..draws {
            counts[rng.below(6)] += 1;
        }
        for c in counts {
            assert!((c as f64 - 100_000.0).abs() < 1_500.0, "{counts:?}");
        }
        let mean: f64 = (0..100_000).map(|_| rng.next_f64()).sum::<f64>() / 100_000.0;
        assert!((mean - 0.5).abs() < 0.005, "{mean}");
        assert!((0..10_000).all(|_| (0.0..1.0).contains(&rng.next_f64())));
    }
}
