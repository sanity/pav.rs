use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use serde::Serialize;
use thiserror::Error;

/// Trait for coordinate types used in Point
pub trait Coordinate:
    Copy + Clone + PartialOrd + PartialEq +
    Add<Output = Self> + Sub<Output = Self> + Mul<Self, Output = Self> + Div<Self, Output = Self> +
    AddAssign + Neg<Output = Self>
{
    /// Returns the zero value for this coordinate type
    fn zero() -> Self;

    /// Returns the one value for this coordinate type
    fn one() -> Self;

    /// Converts the coordinate to a float representation
    fn to_float(&self) -> f64;

    /// Creates a coordinate from a float representation
    fn from_float(value: f64) -> Self;

    /// Computes the absolute difference between two coordinates
    fn abs_diff(&self, other: &Self) -> Self;

    /// Checks if the coordinate is less than zero
    fn is_negative(&self) -> bool;

    /// Computes the average of two coordinates
    fn average(&self, other: &Self) -> Self;
}

impl Coordinate for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn to_float(&self) -> f64 {
        *self
    }

    fn from_float(value: f64) -> Self {
        value
    }

    fn abs_diff(&self, other: &Self) -> Self {
        (self - other).abs()
    }

    fn is_negative(&self) -> bool {
        *self < 0.0
    }

    fn average(&self, other: &Self) -> Self {
        (self + other) / 2.0
    }
}

/// Errors that can occur during isotonic regression
#[derive(Error, Debug)]
pub enum IsotonicRegressionError {
    /// Error when a negative point is encountered with intersect_origin set to true
    #[error("With intersect_origin = true, all points must be >= 0 on both x and y axes")]
    NegativePointWithIntersectOrigin,
}

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.

#[derive(Debug, Clone, Serialize)]
pub struct IsotonicRegression<T: Coordinate> {
    direction: Direction,
    points: Vec<Point<T>>,
    centroid_point: Centroid<T>,
    intersect_origin: bool,
}

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone, Serialize)]
pub struct Point<T: Coordinate> {
    x: T,
    y: T,
    weight: T,
}

impl<T: Coordinate> Default for Point<T> {
    fn default() -> Self {
        Point {
            x: T::zero(),
            y: T::zero(),
            weight: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Centroid<T: Coordinate> {
    sum_x: T,
    sum_y: T,
    sum_weight: T,
}

#[derive(Debug, Clone, Serialize)]
enum Direction {
    Ascending,
    Descending,
}

impl<T: Coordinate + Display> Display for IsotonicRegression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "IsotonicRegression {{")?;
        writeln!(f, "\tdirection: {:?},", self.direction)?;
        writeln!(f, "\tpoints:")?;
        for point in &self.points {
            writeln!(f, "\t\t{}\t{:.2}\t{:.2}", point.x, point.y, point.weight)?;
        }
        writeln!(f, "\tcentroid_point:")?;
        writeln!(
            f,
            "\t\t{}\t{:.2}\t{:.2}",
            self.centroid_point.sum_x, self.centroid_point.sum_y, self.centroid_point.sum_weight
        )?;
        write!(f, "}}")
    }
}

impl<T: Coordinate> IsotonicRegression<T> {
    /// Find an ascending isotonic regression from a set of points
    pub fn new_ascending(points: &[Point<T>]) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Ascending, false)
    }

    /// Find a descending isotonic regression from a set of points
    pub fn new_descending(points: &[Point<T>]) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Descending, false)
    }

    /// Find an isotonic regression in the specified direction. If `intersect_origin` is true, the
    /// regression will intersect the origin (0,0) and all points must be >= 0 on both axes.
    fn new(points: &[Point<T>], direction: Direction, intersect_origin: bool) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        let (sum_x, sum_y, sum_weight) = points.iter().try_fold((T::zero(), T::zero(), T::zero()), |(sx, sy, sw), point| {
            if intersect_origin && (point.x.is_negative() || point.y.is_negative()) {
                Err(IsotonicRegressionError::NegativePointWithIntersectOrigin)
            } else {
                Ok((sx + point.x * point.weight, sy + point.y * point.weight, sw + point.weight))
            }
        })?;

        Ok(IsotonicRegression {
            direction: direction.clone(),
            points: isotonic(points, direction),
            centroid_point: Centroid {
                sum_x,
                sum_y,
                sum_weight,
            },
            intersect_origin,
        })
    }

    /// Find the _y_ point at position `at_x` or None if the regression is empty
    #[must_use]
    pub fn interpolate(&self, at_x: T) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }

        let interpolation = if self.points.len() == 1 {
            self.points[0].y
        } else {
            let pos = self
                .points
                .binary_search_by(|p| p.x.partial_cmp(&at_x).unwrap());
            match pos {
                Ok(ix) => self.points[ix].y,
                Err(ix) => {
                    if ix < 1 {
                        if self.intersect_origin {
                            interpolate_two_points(
                                &Point::new(T::zero(), T::zero()),
                                self.points.first().unwrap(),
                                at_x,
                            )
                        } else {
                            interpolate_two_points(
                                self.points.first().unwrap(),
                                &self.get_centroid_point().unwrap(),
                                at_x,
                            )
                        }
                    } else if ix >= self.points.len() {
                        interpolate_two_points(
                            &self.get_centroid_point().unwrap(),
                            self.points.last().unwrap(),
                            at_x,
                        )
                    } else {
                        interpolate_two_points(&self.points[ix - 1], &self.points[ix], at_x)
                    }
                }
            }
        };

        Some(interpolation)
    }

    /// Retrieve the points that make up the isotonic regression
    pub fn get_points(&self) -> &[Point<T>] {
        &self.points
    }

    /// Retrieve the mean point of the original point set
    pub fn get_centroid_point(&self) -> Option<Point<T>> {
        if self.centroid_point.sum_weight == T::zero() {
            None
        } else {
            Some(Point {
                x: self.centroid_point.sum_x / self.centroid_point.sum_weight,
                y: self.centroid_point.sum_y / self.centroid_point.sum_weight,
                weight: T::one(),
            })
        }
    }

    /// Add new points to the regression
    pub fn add_points(&mut self, points: &[Point<T>]) {
        for point in points {
            assert!(!self.intersect_origin || 
                (!point.x.is_negative() && !point.y.is_negative()), "With intersect_origin = true, all points must be >= 0 on both x and y axes" );
            self.centroid_point.sum_x = self.centroid_point.sum_x + point.x * point.weight;
            self.centroid_point.sum_y = self.centroid_point.sum_y + point.y * point.weight;
            self.centroid_point.sum_weight = self.centroid_point.sum_weight + point.weight;
        }

        let mut new_points = self.points.clone();
        new_points.extend_from_slice(points);
        self.points = isotonic(&new_points, self.direction.clone());
    }

    /// Remove points by inverting their weight and adding
    pub fn remove_points(&mut self, points: &[Point<T>]) {
        self.add_points(
            &points
                .iter()
                .map(|p| Point::new_with_weight(p.x, p.y, -p.weight))
                .collect::<Vec<_>>(),
        );
    }

    /// How many points?
    pub fn len(&self) -> usize {
        self.centroid_point.sum_weight.round() as usize
    }

    /// Are there any points?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroid_point.sum_weight == 0.0
    }
}

impl<T: Coordinate> Point<T> {
    /// Create a new Point
    pub fn new(x: T, y: T) -> Point<T> {
        Point { x, y, weight: T::one() }
    }

    /// Create a new Point with a specified weight
    pub fn new_with_weight(x: T, y: T, weight: T) -> Point<T> {
        Point { x, y, weight }
    }

    // Use getters because modifying points that are part of a regression will have unpredictable
    // results.

    /// The x position of the point
    pub fn x(&self) -> &T {
        &self.x
    }

    /// The y position of the point
    pub fn y(&self) -> &T {
        &self.y
    }

    /// The weight of the point (initially 1.0)
    pub fn weight(&self) -> &T {
        &self.weight
    }

    fn merge_with(&mut self, other: &Point<T>) {
        let total_weight = self.weight + other.weight;
        self.x = (self.x * self.weight + other.x * other.weight) / total_weight;
        self.y = (self.y * self.weight + other.y * other.weight) / total_weight;
        self.weight = total_weight;
    }
}

impl<T: Coordinate> From<(T, T)> for Point<T> {
    fn from(tuple: (T, T)) -> Self {
        Point::new(tuple.0, tuple.1)
    }
}

fn interpolate_two_points<T: Coordinate>(a: &Point<T>, b: &Point<T>, at_x: T) -> T {
    let prop = (at_x - a.x) / (b.x - a.x);
    a.y + (b.y - a.y) * prop
}

fn isotonic<T: Coordinate>(points: &[Point<T>], direction: Direction) -> Vec<Point<T>> {
    let mut merged_points: Vec<Point<T>> = match direction {
        Direction::Ascending => points.to_vec(),
        Direction::Descending => points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };

    // Sort the points by x, and if x is equal, sort by y descending to ensure that points with the same x
    // get merged.
    merged_points.sort_by(|a, b| {
        a.x.partial_cmp(&b.x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.y.partial_cmp(&a.y).unwrap_or(std::cmp::Ordering::Equal))
    });

    let iso_points = merged_points.into_iter().fold(Vec::new(), |mut acc: Vec<Point<T>>, mut point| {
        while let Some(last) = acc.last() {
            if last.y >= point.y {
                point.merge_with(&acc.pop().unwrap());
            } else {
                break;
            }
        }
        acc.push(point);
        acc
    });

    match direction {
        Direction::Ascending => iso_points,
        Direction::Descending => iso_points.into_iter().map(|p| Point { y: -p.y, ..p }).collect(),
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;

    #[test]
    fn usage_example() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        assert_eq!(regression.interpolate(1.5).unwrap(), 1.75);
    }

    #[test]
    fn isotonic_no_points() {
        assert!(isotonic::<f64>(&[], Direction::Ascending).is_empty());
    }

    #[test]
    fn isotonic_one_point() {
        assert_eq!(
            isotonic(&[Point::new(1.0, 2.0)], Direction::Ascending)
                .pop()
                .unwrap(),
            Point::new(1.0, 2.0)
        );
    }

    #[test]
    fn isotonic_simple_merge() {
        assert_eq!(
            isotonic(
                &[Point::new(1.0, 2.0), Point::new(2.0, 0.0)],
                Direction::Ascending
            )
            .pop()
            .unwrap(),
            Point::new_with_weight(1.5, 1.0, 2.0)
        );
    }

    #[test]
    fn isotonic_one_not_merged() {
        assert_eq!(
            isotonic(
                &[
                    Point::new(0.5, -0.5),
                    Point::new(1.0, 2.0),
                    Point::new(2.0, 0.0),
                ],
                Direction::Ascending
            ),
            [Point::new(0.5, -0.5), Point::new_with_weight(1.5, 1.0, 2.0)]
        );
    }

    #[test]
    fn isotonic_merge_three() {
        assert_eq!(
            isotonic(
                &[
                    Point::new(0.0, 1.0),
                    Point::new(1.0, 2.0),
                    Point::new(2.0, -1.0),
                ],
                Direction::Ascending
            ),
            [Point::new_with_weight(1.0, 2.0 / 3.0, 3.0)]
        );
    }

    #[test]
    fn test_interpolate() {
        let regression =
            IsotonicRegression::new_ascending(&[Point::new(1.0, 5.0), Point::new(2.0, 7.0)]).unwrap();
        assert!(regression.interpolate(1.5).unwrap().abs_diff(&6.0) < f64::EPSILON);
    }

    #[test]
    fn test_isotonic_ascending() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, -1.0),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(
                (0.0 + 1.0 + 2.0) / 3.0,
                (1.0 + 2.0 - 1.0) / 3.0,
                3.0
            )]
        )
    }

    #[test]
    fn test_isotonic_descending() {
        let points = &[
            Point::new(0.0, -1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(points).unwrap();
        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(1.0, 2.0 / 3.0, 3.0)]
        )
    }

    #[test]
    fn test_descending_interpolation() {
        let regression = IsotonicRegression::new_descending(&[
            Point::new(0.0, 3.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.0),
        ]).unwrap();
        assert_eq!(regression.interpolate(0.5).unwrap(), 2.5);
    }

    #[test]
    fn test_single_point_regression() {
        let regression = IsotonicRegression::new_ascending(&[Point::new(1.0, 3.0)]).unwrap();
        assert_eq!(regression.interpolate(0.0).unwrap(), 3.0);
    }

    #[test]
    fn test_point_accessors() {
        let point = Point {
            x: 1.0,
            y: 2.0,
            weight: 3.0,
        };
        assert_eq!(*point.x(), 1.0);
        assert_eq!(*point.y(), 2.0);
        assert_eq!(point.weight(), 3.0);
    }

    #[test]
    fn test_add_points_1() {
        let points = &[Point::new(0.0, 1.0)];

        let mut regression = IsotonicRegression::new_ascending(points).unwrap();

        regression.add_points(&[Point::new(1.0, 2.0)]);

        assert_eq!(
            regression.get_points(),
            &[
                Point::new_with_weight(0.0, 1.0, 1.0),
                Point::new_with_weight(1.0, 2.0, 1.0),
            ]
        );
    }

    #[test]
    fn test_add_points_2() {
        let points = &[Point::new(1.0, 2.0)];

        let mut regression = IsotonicRegression::new_ascending(points).unwrap();

        regression.add_points(&[Point::new(0.0, 3.0)]);

        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(0.5, 2.5, 2.0),]
        );
    }

    #[test]
    fn test_add_equal_x() {
        let points = &[Point::new(0.0, 1.0)];

        let mut regression = IsotonicRegression::new_ascending(points).unwrap();

        regression.add_points(&[Point::new(0.0, 2.0)]);

        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(0.0, 1.5, 2.0)]
        );
    }

    // This test generates 100 points at random, it then creates a regression with these points.
    // It then creates a second regression with 50 points, and adds the other 50 points to it.
    // It then checks that the two regression centroids are the same.
    #[test]
    fn test_add_points_random_centroids() {
        let mut rng = rand::thread_rng();
        let mut points = Vec::new();
        for _ in 0..100 {
            points.push(Point::new(
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ));
        }

        let regression = IsotonicRegression::new_ascending(&points).unwrap();

        let mut regression2 = IsotonicRegression::new_ascending(&points[0..(points.len() / 2)]).unwrap();

        regression2.add_points(&points[(points.len() / 2)..points.len()]);

        assert_eq!(regression.centroid_point, regression2.centroid_point);
    }

    // This test creates two IsotonicRegressions, one of which starts empty but with points added,
    // the other of which starts with all the points. It then checks that the two regressions
    // are similar.
    #[test]
    fn test_add_points_random_regression() {
        let mut rng = rand::thread_rng();
        let mut points = Vec::new();
        for _ in 0..100 {
            points.push(Point::new(
                rng.gen_range(0.0..100.0),
                rng.gen_range(0.0..100.0),
            ));
        }

        let regression = IsotonicRegression::new_ascending(&points).unwrap();

        let mut regression2 = IsotonicRegression::new_ascending(&[]).unwrap();

        assert_eq!(regression2.get_points(), &[]);

        regression2.add_points(&points);

        assert_eq!(regression.centroid_point, regression2.centroid_point);
    }

    #[test]
    fn test_add_points_panic() {
        let regression = IsotonicRegression::new_ascending(&[]).unwrap();

        assert_eq!(regression.interpolate(50.0), None);
    }
}
