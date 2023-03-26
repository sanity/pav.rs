use ordered_float::OrderedFloat;

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.

#[derive(Debug, Clone)]
pub struct IsotonicRegression {
    points: Vec<Point>,
    centroid_point: Point,
}

enum Direction {
    Ascending,
    Descending,
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
        assert!(!points.is_empty(), "points is empty, can't create regression");
        let point_count: f64 = points.iter().map(|p| p.weight).sum();
        let mut sum_x: f64 = 0.0;
        let mut sum_y: f64 = 0.0;
        for point in points {
            sum_x += point.x * point.weight;
            sum_y += point.y * point.weight;
        }

        IsotonicRegression {
            points: isotonic(points, direction),
            centroid_point: Point::new(sum_x / point_count, sum_y / point_count),
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
            return match pos {
                Ok(ix) => self.points[ix].y,
                Err(ix) => {
                    if ix < 1 {
                        interpolate_two_points(
                            self.points.first().unwrap(),
                            &self.centroid_point,
                            &at_x,
                        )
                    } else if ix >= self.points.len() {
                        interpolate_two_points(
                            &self.centroid_point,
                            self.points.last().unwrap(),
                            &at_x,
                        )
                    } else {
                        interpolate_two_points(&self.points[ix - 1], &self.points[ix], &at_x)
                    }
                }
            };
        }
    }

    /// Retrieve the points that make up the isotonic regression
    pub fn get_points(&self) -> &[Point] {
        &self.points
    }

    /// Retrieve the mean point of the original point set
    pub fn get_centroid_point(&self) -> &Point {
        &self.centroid_point
    }
}

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone)]
pub struct Point {
    x: f64,
    y: f64,
    weight: f64,
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

fn interpolate_two_points(a: &Point, b: &Point, at_x: &f64) -> f64 {
    let prop = (at_x - (a.x)) / (b.x - a.x);
    (b.y - a.y) * prop + a.y
}

fn isotonic(points: &[Point], direction: Direction) -> Vec<Point> {
    let mut merged_points: Vec<Point> = match direction {
        Direction::Ascending => points.to_vec(),
        Direction::Descending => points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };

    merged_points.sort_by_key(|point| OrderedFloat(point.x));

    let mut iso_points: Vec<Point> = Vec::new();
    for point in &mut merged_points.iter() {
        if iso_points.is_empty() || (point.y > iso_points.last().unwrap().y) {
            iso_points.push(*point)
        } else {
            let mut new_point = *point;
            loop {
                if iso_points.is_empty() || (iso_points.last().unwrap().y < (new_point).y) {
                    iso_points.push(new_point);
                    break;
                } else {
                    let last_to_repl = iso_points.pop();
                    new_point.merge_with(&last_to_repl.unwrap());
                }
            }
        }
    }

    return match direction {
        Direction::Ascending => iso_points,
        Direction::Descending => iso_points.iter().map(|p| Point { y: -p.y, ..*p }).collect(),
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usage_example() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
        ];

        let regression = IsotonicRegression::new_ascending(points);
        assert_eq!(
            regression.interpolate(1.5), 1.75
        );
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
        let regression = IsotonicRegression::new_ascending(&[
            Point::new(1.0, 3.0),
        ]);
        assert_eq!(regression.interpolate(0.0), 3.0);
    }

    #[test]
    fn test_point_accessors() {
        let point = Point { x: 1.0, y: 2.0 , weight : 3.0};
        assert_eq!(point.x(), 1.0);
        assert_eq!(point.y(), 2.0);
        assert_eq!(point.weight(), 3.0);
    }
}
