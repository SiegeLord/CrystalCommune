use crate::sprite;
use allegro::*;
use na::Point2;
use nalgebra as na;
use rand::prelude::*;

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
	pub thought: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SceneryDraw
{
	pub sprite: String,
	pub variant: i32,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ProviderKind
{
	TakenHouse(hecs::Entity),
	EmptyHouse,
	Mine,
	Port,
	Cafe,
	Plot1x1,
	Plot3x2,
	Plot3x3,
	Road,
}

impl ProviderKind
{
	pub fn get_size(&self) -> (i32, i32)
	{
		match self
		{
			ProviderKind::TakenHouse(_) => (3, 2),
			ProviderKind::EmptyHouse => (3, 2),
			ProviderKind::Mine => (3, 3),
			ProviderKind::Port => (3, 3),
			ProviderKind::Cafe => (3, 2),
			ProviderKind::Plot1x1 => (1, 1),
			ProviderKind::Plot3x2 => (3, 2),
			ProviderKind::Plot3x3 => (3, 3),
			ProviderKind::Road => (1, 1),
		}
	}

	pub fn get_max_occupants(&self) -> i32
	{
		match self
		{
			ProviderKind::TakenHouse(_) => 1,
			ProviderKind::EmptyHouse => 1,
			ProviderKind::Mine => 3,
			ProviderKind::Port => 5,
			ProviderKind::Cafe => 3,
			ProviderKind::Plot1x1 => 2,
			ProviderKind::Plot3x2 => 2,
			ProviderKind::Plot3x3 => 2,
			ProviderKind::Road => 1,
		}
	}

	pub fn get_sprite<T: Rng>(&self, rng: &mut T) -> (&'static str, i32)
	{
		match self
		{
			ProviderKind::EmptyHouse => (
				["data/house1.cfg", "data/house2.cfg"].choose(rng).unwrap(),
				1,
			),
			ProviderKind::TakenHouse(_) => ("data/house1.cfg", 0),
			ProviderKind::Mine => ("data/mine1.cfg", 0),
			ProviderKind::Port => ("data/port.cfg", 0),
			ProviderKind::Cafe => ("data/cafe.cfg", 0),
			ProviderKind::Plot1x1 => ("data/plot_1x1.cfg", 0),
			ProviderKind::Plot3x2 => ("data/plot_3x2.cfg", 0),
			ProviderKind::Plot3x3 => ("data/plot_3x3.cfg", 0),
			ProviderKind::Road => ("data/road.cfg", 0),
		}
	}

	pub fn get_cost(&self) -> i32
	{
		match self
		{
			ProviderKind::TakenHouse(_) => 0,
			ProviderKind::EmptyHouse => 500,
			ProviderKind::Mine => 1500,
			ProviderKind::Port => 3000,
			ProviderKind::Cafe => 1000,
			ProviderKind::Road => 100,
			ProviderKind::Plot1x1 => 0,
			ProviderKind::Plot3x2 => 0,
			ProviderKind::Plot3x3 => 0,
		}
	}
	pub fn get_work_total(&self) -> i32
	{
		match self
		{
			ProviderKind::TakenHouse(_) => 0,
			ProviderKind::EmptyHouse => 2,
			ProviderKind::Mine => 5,
			ProviderKind::Port => 5,
			ProviderKind::Cafe => 3,
			ProviderKind::Road => 1,
			ProviderKind::Plot1x1 => 0,
			ProviderKind::Plot3x2 => 0,
			ProviderKind::Plot3x3 => 0,
		}
	}
}

#[derive(Debug, Clone)]
pub struct Provider
{
	pub kind: ProviderKind,
	pub num_occupants: i32,
	pub max_occupants: i32,
	pub time_to_maintain: f64,
}

#[derive(Debug, Clone)]
pub struct Plot
{
	pub kind: ProviderKind,
	pub work_done: i32,
	pub work_total: i32,
}

#[derive(Debug, Clone)]
pub struct Agent
{
	pub time_to_think: f64,
	pub time_to_work: f64,
	pub cur_provider: Option<hecs::Entity>,
	pub house: Option<hecs::Entity>,
	pub leaving: bool,
	pub sleepyness: i32,
	pub hunger: i32,
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

#[derive(Debug, Clone)]
pub struct Indicator
{
	pub time_to_die: f64,
	pub text: String,
	pub color: Color,
}

#[derive(Debug, Clone)]
pub struct Crystal;
