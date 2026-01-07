use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Dot,
}

impl DistanceMetric {
    pub(crate) fn score(&self, left: &[f32], right: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => euclidean_distance(left, right),
            DistanceMetric::Cosine => cosine_distance(left, right),
            DistanceMetric::Dot => dot_distance(left, right),
        }
    }

    pub(crate) fn to_u8(self) -> u8 {
        match self {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::Cosine => 1,
            DistanceMetric::Dot => 2,
        }
    }

    pub(crate) fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(DistanceMetric::Euclidean),
            1 => Some(DistanceMetric::Cosine),
            2 => Some(DistanceMetric::Dot),
            _ => None,
        }
    }
}

pub(crate) fn euclidean_distance(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(left: &[f32], right: &[f32]) -> f32 {
    let denom = l2_norm(left) * l2_norm(right);
    if denom == 0.0 {
        return f32::INFINITY;
    }
    let similarity = dot_product(left, right) / denom;
    1.0 - similarity
}

fn dot_distance(left: &[f32], right: &[f32]) -> f32 {
    -dot_product(left, right)
}

fn dot_product(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}
