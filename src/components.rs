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
	pub flies: bool,
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
	Office,
	Port,
    Plot3x2,
    Plot3x3,
}

impl ProviderKind
{
	pub fn get_size(&self) -> (i32, i32)
	{
		match self
		{
			ProviderKind::TakenHouse(_) => (3, 2),
			ProviderKind::EmptyHouse => (3, 2),
			ProviderKind::Office => (3, 3),
			ProviderKind::Port => (3, 3),
            ProviderKind::Plot3x2 => (3, 2),
            ProviderKind::Plot3x3 => (3, 3),
		}
	}

	pub fn get_max_occupants(&self) -> i32
	{
		match self
		{
			ProviderKind::TakenHouse(_) => 1,
			ProviderKind::EmptyHouse => 1,
			ProviderKind::Office => 3,
			ProviderKind::Port => 5,
            ProviderKind::Plot3x2 => 2,
            ProviderKind::Plot3x3 => 2,
		}
	}

	pub fn get_sprite(&self) -> &'static str
	{
		match self
		{
			ProviderKind::EmptyHouse => "data/empty_house1.cfg",
			ProviderKind::TakenHouse(_) => "data/house1.cfg",
			ProviderKind::Office => "data/office1.cfg",
			ProviderKind::Port => "data/port.cfg",
            ProviderKind::Plot3x2 => "data/plot_3x2.cfg",
            ProviderKind::Plot3x3 => "data/plot_3x3.cfg",
		}
	}
}

#[derive(Debug, Clone)]
pub struct Provider
{
	pub kind: ProviderKind,
	pub num_occupants: i32,
	pub max_occupants: i32,
}

#[derive(Debug, Clone)]
pub struct Plot
{
    pub kind: ProviderKind,
    pub work_left: i32,
}

#[derive(Debug, Clone)]
pub struct Agent
{
	pub time_to_think: f64,
	pub time_to_work: f64,
	pub cur_provider: Option<hecs::Entity>,
	pub house: Option<hecs::Entity>,
	pub leaving: bool,
}

#[derive(Debug, Clone)]
pub struct BuildingPlacement
{
	pub width: i32,
	pub height: i32,
	pub valid: Vec<bool>,
	pub kind: ProviderKind,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BlimpState
{
	Spawned,
	Arriving,
	Waiting,
	Leaving,
}

#[derive(Debug, Clone)]
pub struct Blimp
{
	pub state: BlimpState,
	pub time_to_leave: f64,
}
