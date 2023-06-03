use crate::sprite;
use na::Point2;
use nalgebra as na;

#[derive(Debug, Copy, Clone)]
pub struct Position
{
	pub pos: Point2<i32>,
	pub dir: i32,
}

#[derive(Debug, Copy, Clone)]
pub struct CanMove
{
    pub moving: bool,
}

#[derive(Debug, Clone)]
pub struct TilePath
{
    pub tile_path: Vec<Point2<i32>>,
}

#[derive(Debug, Clone)]
pub struct AgentDraw
{
    pub sprite: String,
}

