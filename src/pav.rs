use std::fmt::{Display, Formatter};

use ordered_float::OrderedFloat;
use serde::Serialize;

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.

#[derive(Debug, Clone, Serialize)]
pub struct IsotonicRegression {
    direction: Direction,
    points: Vec<Point>,
    centroid_point: Centroid,
}

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone, Serialize)]
pub struct Point {
    x: f64,
    y: f64,
    weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Centroid {
    sum_x: f64,
    sum_y: f64,
    sum_weight: f64,
}

#[derive(Debug, Clone, Serialize)]
enum Direction {
    Ascending,
    Descending,
}

impl Display for IsotonicRegression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "IsotonicRegression {{")?;
        writeln!(f, "\tdirection: {:?},", self.direction)?;
        writeln!(f, "\tpoints:")?;
        for point in &self.points {
            writeln!(f, "\t\t{:.2}\t{:.2}\t{:.2}", point.x, point.y, point.weight)?;
        }
        writeln!(f, "\tcentroid_point:")?;
        writeln!(
            f,
            "\t\t{:.2}\t{:.2}\t{:.2}",
            self.centroid_point.sum_x, self.centroid_point.sum_y, self.centroid_point.sum_weight
        )?;
        write!(f, "}}")
    }
}

impl IsotonicRegression {
    /// Find an ascending isotonic regression from a set of points
    pub fn new_ascending(points: &[Point]) -> IsotonicRegression {
        IsotonicRegression::new(points, Direction::Ascending)
    }

    /// Find a descending isotonic regression from a set of points
    pub fn new_descending(points: &[Point]) -> IsotonicRegression {
        IsotonicRegression::new(points, Direction::Descending)
    }

    fn new(points: &[Point], direction: Direction) -> IsotonicRegression {
        assert!(
            !points.is_empty(),
            "points is empty, can't create regression"
        );
        let point_count: f64 = points.iter().map(Point::weight).sum();
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        for point in points {
            sum_x += point.x * point.weight;
            sum_y += point.y * point.weight;
        }

        IsotonicRegression {
            direction: direction.clone(),
            points: isotonic(points, direction),
            centroid_point: Centroid {
                sum_x,
                sum_y,
                sum_weight: point_count,
            },
        }
    }

    /// Find the _y_ point at position `at_x`
    pub fn interpolate(&self, at_x: f64) -> f64 {
        if self.points.len() == 1 {
            self.points[0].y
        } else {
            let pos = self
                .points
                .binary_search_by_key(&OrderedFloat(at_x), |p| OrderedFloat(p.x));
            match pos {
                Ok(ix) => self.points[ix].y,
                Err(ix) => {
                    if ix < 1 {
                        interpolate_two_points(
                            self.points.first().unwrap(),
                            &self.get_centroid_point(),
                            at_x,
                        )
                    } else if ix >= self.points.len() {
                        interpolate_two_points(
                            &self.get_centroid_point(),
                            self.points.last().unwrap(),
                            at_x,
                        )
                    } else {
                        interpolate_two_points(&self.points[ix - 1], &self.points[ix], at_x)
                    }
                }
            }
        }
    }

    /// Retrieve the points that make up the isotonic regression
    pub fn get_points(&self) -> &[Point] {
        &self.points
    }

    /// Retrieve the mean point of the original point set
    pub fn get_centroid_point(&self) -> Point {
        Point {
            x: self.centroid_point.sum_x / self.centroid_point.sum_weight,
            y: self.centroid_point.sum_y / self.centroid_point.sum_weight,
            weight: 1.0,
        }
    }

    /// Add new points to the regression
    pub fn add_points(&mut self, points: &[Point]) {
        for point in points {
            self.centroid_point.sum_x += point.x * point.weight;
            self.centroid_point.sum_y += point.y * point.weight;
            self.centroid_point.sum_weight += point.weight;
        }

        let mut new_points = self.points.clone();
        new_points.extend(points);
        self.points = isotonic(&new_points, self.direction.clone());
    }

    /// Remove points by inverting their weight and adding
    pub fn remove_points(&mut self, points: &[Point]) {
        self.add_points(
            points
                .iter()
                .map(|p| Point::new_with_weight(p.x, p.y, -p.weight))
                .collect::<Vec<_>>()
                .as_slice(),
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

impl Point {
    /// Create a new Point
    pub fn new(x: f64, y: f64) -> Point {
        Point { x, y, weight: 1.0 }
    }

    /// Create a new Point with a specified weight
    pub fn new_with_weight(x: f64, y: f64, weight: f64) -> Point {
        Point { x, y, weight }
    }

    // Use getters because modifying points that are part of a regression will have unpredictable
    // results.

    /// The x position of the point
    pub fn x(&self) -> f64 {
        self.x
    }

    /// The y position of the point
    pub fn y(&self) -> f64 {
        self.y
    }

    /// The weight of the point (initially 1.0)
    pub fn weight(&self) -> f64 {
        self.weight
    }

    fn merge_with(&mut self, other: &Point) {
        self.x = ((self.x * self.weight) + (other.x * other.weight)) / (self.weight + other.weight);

        self.y = ((self.y * self.weight) + (other.y * other.weight)) / (self.weight + other.weight);

        self.weight += other.weight;
    }
}

fn interpolate_two_points(a: &Point, b: &Point, at_x: f64) -> f64 {
    let prop = (at_x - (a.x)) / (b.x - a.x);
    (b.y - a.y) * prop + a.y
}

fn isotonic(points: &[Point], direction: Direction) -> Vec<Point> {
    let mut merged_points: Vec<Point> = match direction {
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
    
    let mut iso_points: Vec<Point> = Vec::new();
    for point in merged_points.iter() {
        let mut new_point = *point;
        while let Some(last_point) = iso_points.last() {
            if last_point.y < new_point.y {
                break;
            }
            let last_to_repl = iso_points.pop().unwrap();
            new_point.merge_with(&last_to_repl);
        }
        iso_points.push(new_point);
    }

    return match direction {
        Direction::Ascending => iso_points,
        Direction::Descending => iso_points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };
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

        let regression = IsotonicRegression::new_ascending(points);
        assert_eq!(regression.interpolate(1.5), 1.75);
    }

    #[test]
    fn isotonic_no_points() {
        assert!(isotonic(&[], Direction::Ascending).is_empty());
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
            IsotonicRegression::new_ascending(&[Point::new(1.0, 5.0), Point::new(2.0, 7.0)]);
        assert!((regression.interpolate(1.5) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_isotonic_ascending() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, -1.0),
        ];

        let regression = IsotonicRegression::new_ascending(points);
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
        let regression = IsotonicRegression::new_descending(points);
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
        ]);
        assert_eq!(regression.interpolate(0.5), 2.5);
    }

    #[test]
    fn test_single_point_regression() {
        let regression = IsotonicRegression::new_ascending(&[Point::new(1.0, 3.0)]);
        assert_eq!(regression.interpolate(0.0), 3.0);
    }

    #[test]
    fn test_point_accessors() {
        let point = Point {
            x: 1.0,
            y: 2.0,
            weight: 3.0,
        };
        assert_eq!(point.x(), 1.0);
        assert_eq!(point.y(), 2.0);
        assert_eq!(point.weight(), 3.0);
    }

    #[test]
    fn test_add_points_1() {
        let points = &[Point::new(0.0, 1.0)];

        let mut regression = IsotonicRegression::new_ascending(points);

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

        let mut regression = IsotonicRegression::new_ascending(points);

        regression.add_points(&[Point::new(0.0, 3.0)]);

        assert_eq!(
            regression.get_points(),
            &[Point::new_with_weight(0.5, 2.5, 2.0),]
        );
    }

    #[test]
    fn test_add_equal_x() {
        let points = &[Point::new(0.0, 1.0)];

        let mut regression = IsotonicRegression::new_ascending(points);

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

        let regression = IsotonicRegression::new_ascending(&points);

        let mut regression2 = IsotonicRegression::new_ascending(&points[0..(points.len() / 2)]);

        regression2.add_points(&points[(points.len() / 2)..points.len()]);

        assert_eq!(regression.centroid_point, regression2.centroid_point);
    }
}
