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
    pub visible: bool,
}

#[derive(Debug, Clone)]
pub struct SceneryDraw
{
	pub sprite: String,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ProviderKind
{
    TakenHouse(hecs::Entity),
	EmptyHouse,
	Work,
}

#[derive(Debug, Clone)]
pub struct Provider
{
	pub kind: ProviderKind,
    pub num_occupants: i32,
    pub max_occupants: i32,
}

#[derive(Debug, Clone)]
pub struct Agent
{
	pub time_to_think: f64,
    pub time_to_work: f64,
    pub cur_provider: Option<hecs::Entity>,
    pub house: Option<hecs::Entity>,
}
