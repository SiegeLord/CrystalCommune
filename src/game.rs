use crate::error::Result;
use crate::{astar, components as comps, controls, game_state, sprite, utils};
use allegro::*;
use allegro_font::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

use std::collections::HashMap;

static TILE_SIZE: i32 = 16;
static MAX_SLEEPYNESS: i32 = 3000;
static MAX_HUNGER: i32 = 3000;
static NUM_BUTTONS: i32 = 10;
static MEAN_TICKS: i32 = 500;

fn diamond_square<R: Rng>(size: i32, rng: &mut R) -> Vec<i32>
{
	assert!(size >= 0);
	let real_size = 2i32.pow(size as u32) + 1;
	dbg!(real_size);

	let global_max_height = 8;

	let mut heightmap = vec![-1i32; (real_size * real_size) as usize];

	//~ for stage in 0..=2
	for stage in 0..=size
	{
		let num_cells = 2i32.pow(stage as u32);
		let spacing = (real_size - 1) / num_cells;
		//~ dbg!(stage);
		//~ dbg!(spacing);

		// Square
		for y_idx in 0..=num_cells
		{
			for x_idx in 0..=num_cells
			{
				let y = y_idx * spacing;
				let x = x_idx * spacing;
				if heightmap[(x + y * real_size) as usize] == -1
				{
					let mut min_height = 0;
					let mut max_height = global_max_height;
					let mut mean_height = 0.;
					let mut count = 0;

					//~ println!();

					// Check the diag corners
					for sy in [-1, 1]
					{
						for sx in [-1, 1]
						{
							let cx = x + sx * spacing;
							let cy = y + sy * spacing;
							if cx >= 0 && cy >= 0 && cx < real_size && cy < real_size
							{
								let val = heightmap[(cx + cy * real_size) as usize];
								if val >= 0
								{
									min_height = utils::max(min_height, val - spacing);
									max_height = utils::min(max_height, val + spacing);
								}
							}
						}
					}

					// Check the rect corners
					for [sx, sy] in [[-1, 0], [0, -1], [1, 0], [0, 1]]
					{
						let cx = x + sx * spacing;
						let cy = y + sy * spacing;
						if cx >= 0 && cy >= 0 && cx < real_size && cy < real_size
						{
							let val = heightmap[(cx + cy * real_size) as usize];
							if val >= 0
							{
								min_height = utils::max(min_height, val - spacing);
								max_height = utils::min(max_height, val + spacing);

								mean_height =
									(mean_height * count as f32 + val as f32) / (count + 1) as f32;
								count += 1;
							}
						}
					}

					if count > 0
					{
						// TODO: Check this jitter values.
						min_height = utils::max(min_height, mean_height as i32 - 2);
						max_height = utils::min(max_height, mean_height as i32 + 2);
					}

					//~ dbg!(stage, x, y, min_height, max_height);
					let new_val = rng.gen_range(min_height..=max_height);
					//~ dbg!(new_val);
					heightmap[(x + y * real_size) as usize] = new_val;
				}
			}
		}

		// Diamond
		for y_idx in 0..num_cells
		{
			for x_idx in 0..num_cells
			{
				let y = y_idx * spacing + spacing / 2;
				let x = x_idx * spacing + spacing / 2;
				if heightmap[(x + y * real_size) as usize] == -1
				{
					let mut min_height = 0;
					let mut max_height = global_max_height;
					let mut mean_height = 0.;
					let mut count = 0;
					//~ println!();
					// Check the diag corners
					for sy in [-1, 1]
					{
						for sx in [-1, 1]
						{
							let cx = x + sx * spacing / 2;
							let cy = y + sy * spacing / 2;
							if cx >= 0 && cy >= 0 && cx < real_size && cy < real_size
							{
								let val = heightmap[(cx + cy * real_size) as usize];
								if val >= 0
								{
									min_height = utils::max(min_height, val - spacing / 2);
									max_height = utils::min(max_height, val + spacing / 2);

									mean_height = (mean_height * count as f32 + val as f32)
										/ (count + 1) as f32;
									count += 1;
								}
							}
						}
					}

					// Check the rect corners
					for [sx, sy] in [[-1, 0], [0, -1], [1, 0], [0, 1]]
					{
						let cx = x + sx * spacing;
						let cy = y + sy * spacing;
						if cx >= 0 && cy >= 0 && cx < real_size && cy < real_size
						{
							let val = heightmap[(cx + cy * real_size) as usize];
							if val >= 0
							{
								min_height = utils::max(min_height, val - spacing);
								max_height = utils::min(max_height, val + spacing);
							}
						}
					} // 3, 3

					if count > 0
					{
						// TODO: Check this jitter values.
						min_height = utils::max(min_height, mean_height as i32 - 2);
						max_height = utils::min(max_height, mean_height as i32 + 2);
					}
					//~ dbg!(x, y, stage, min_height, max_height);
					let new_val = rng.gen_range(min_height..=max_height);
					//~ dbg!(new_val);
					heightmap[(x + y * real_size) as usize] = new_val;
				}
			}
		}
	}
	heightmap
}

fn smooth_heightmap(heightmap: &[i32]) -> Vec<i32>
{
	let real_size = (heightmap.len() as f32).sqrt() as i32;
	let mut res = vec![0; heightmap.len()];
	for y in 0..real_size
	{
		for x in 0..real_size
		{
			let mut mean_height = 0.;
			let mut count = 0;
			for sy in [-1, 1]
			{
				for sx in [-1, 1]
				{
					let cx = x + sx;
					let cy = y + sy;
					if cx >= 0 && cy >= 0 && cx < real_size && cy < real_size
					{
						let val = heightmap[(cx + cy * real_size) as usize];
						mean_height =
							(mean_height * count as f32 + val as f32) / (count + 1) as f32;
						count += 1;
					}
				}
			}
			res[(x + y * real_size) as usize] = mean_height as i32;
		}
	}
	res
}

fn lower_heightmap(heightmap: &[i32]) -> Vec<i32>
{
	let mut min_height = 1000;
	for v in heightmap
	{
		min_height = utils::min(*v, min_height);
	}
	let mut res = heightmap.to_vec();
	for v in &mut res
	{
		*v -= min_height;
	}
	res
}

pub struct Game
{
	map: Map,
}

impl Game
{
	pub fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		Ok(Self {
			map: Map::new(state)?,
		})
	}

	pub fn logic(
		&mut self, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		self.map.logic(state)
	}

	pub fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		self.map.input(event, state)
	}

	pub fn draw(&mut self, state: &game_state::GameState) -> Result<()>
	{
		state.core.clear_to_color(Color::from_rgb_f(0.5, 0.5, 1.));
		self.map.draw(state)?;
		Ok(())
	}
}

#[rustfmt::skip]
fn decode_tile<T: Rng>(vals: [i32; 4], rng: &mut T) -> i32
{
	match vals
    {
        [1, 1,
         1, 1] => rng.gen_range(1..=2),
        
        [0, 0,
         1, 1] => 4,
        [1, 0,
         1, 0] => 5,
        [1, 1,
         0, 0] => 6,
        [0, 1,
         0, 1] => 7,

        [0, 0,
         0, 1] => 8,
        [0, 0,
         1, 0] => 9,
        [1, 0,
         0, 0] => 10,
        [0, 1,
         0, 0] => 11,

        [1, 0,
         0, 1] => 12,
        [0, 1,
         1, 0] => 13,

        [1, 1,
         1, 0] => 14,
        [1, 1,
         0, 1] => 15,
        [1, 0,
         1, 1] => 16,
        [0, 1,
         1, 1] => 17,
        _ => 0,
    }
}

fn tile_to_idx(tile_pos: Point2<i32>, size: usize) -> Option<usize>
{
	if tile_pos.x < 0 || tile_pos.y < 0 || tile_pos.x >= size as i32 || tile_pos.y >= size as i32
	{
		None
	}
	else
	{
		Some((tile_pos.y * size as i32 + tile_pos.x) as usize)
	}
}

fn tile_to_pixel(tile_pos: Point2<i32>) -> Point2<i32>
{
	tile_pos * TILE_SIZE
}

fn pixel_to_tile(pixel_pos: Point2<i32>) -> Point2<i32>
{
	Point2::new(pixel_pos.x / TILE_SIZE, pixel_pos.y / TILE_SIZE)
}

fn to_f32(pos: Point2<i32>) -> Point2<f32>
{
	Point2::new(pos.x as f32, pos.y as f32)
}

fn get_warp_location<T: Rng>(rng: &mut T, size: usize) -> (i32, i32)
{
	*[
		(0, 0),
		(0, size as i32 - 1),
		(size as i32 - 1, 0),
		(size as i32 - 1, size as i32 - 1),
	]
	.choose(rng)
	.unwrap()
}

fn get_money_indicator(amount: i32, pos_sign: bool) -> (String, Color)
{
	let text = format!(
		"{}${}",
		if amount >= 0
		{
			if pos_sign
			{
				"+"
			}
			else
			{
				""
			}
		}
		else
		{
			"-"
		},
		amount.abs()
	);
	let color = if amount >= 0
	{
		Color::from_rgb(208, 248, 171)
	}
	else
	{
		Color::from_rgb_f(0.8, 0.2, 0.3)
	};
	(text, color)
}

pub fn spawn_indicator(
	tile_pos: Point2<i32>, text: String, color: Color, world: &mut hecs::World,
	state: &mut game_state::GameState,
) -> Result<hecs::Entity>
{
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::Indicator {
			text: text,
			color: color,
			time_to_die: state.time() + 1.0,
		},
	));
	Ok(entity)
}

pub fn spawn_agent<T: Rng>(
	tile_pos: Point2<i32>, world: &mut hecs::World, _state: &mut game_state::GameState, rng: &mut T,
) -> Result<hecs::Entity>
{
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::TilePath { tile_path: vec![] },
		comps::AgentDraw {
			sprite: ["data/cat1.cfg", "data/cat2.cfg", "data/duck.cfg"]
				.choose(rng)
				.unwrap()
				.to_string(),
			visible: true,
			thought: None,
		},
		comps::CanMove {
			moving: false,
			flies: false,
		},
		comps::Agent {
			time_to_think: 0.,
			time_to_work: 0.,
			cur_provider: None,
			house: None,
			leaving: false,
			hunger: 0,
			sleepyness: 0,
		},
	));
	Ok(entity)
}

pub fn spawn_provider<T: Rng>(
	tile_pos: Point2<i32>, mut kind: comps::ProviderKind, world: &mut hecs::World,
	state: &mut game_state::GameState, rng: &mut T,
) -> Result<hecs::Entity>
{
	let size = kind.get_size();
	let real_kind = kind;
	match size
	{
		(1, 1) => kind = comps::ProviderKind::Plot1x1,
		(3, 2) => kind = comps::ProviderKind::Plot3x2,
		(3, 3) => kind = comps::ProviderKind::Plot3x3,
		_ => panic!("Unknown plot size: {:?}", size),
	}

	let (sprite, variant) = kind.get_sprite(rng);
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::SceneryDraw {
			sprite: sprite.to_string(),
			variant: variant,
		},
		comps::Provider {
			kind: kind,
			num_occupants: 0,
			max_occupants: kind.get_max_occupants(),
			time_to_maintain: state.time() + 10.,
		},
		comps::Plot {
			kind: real_kind,
			work_done: 0,
			work_total: real_kind.get_work_total(),
		},
	));
	Ok(entity)
}

pub fn spawn_crystal<T: Rng>(
	tile_pos: Point2<i32>, world: &mut hecs::World, rng: &mut T,
) -> Result<hecs::Entity>
{
	let sprite = ["data/crystal1.cfg", "data/crystal2.cfg"]
		.choose(rng)
		.unwrap();

	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::SceneryDraw {
			sprite: sprite.to_string(),
			variant: 0,
		},
		comps::Crystal,
	));
	Ok(entity)
}

pub fn spawn_building_placement(
	tile_pos: Point2<i32>, kind: comps::ProviderKind, world: &mut hecs::World,
) -> Result<hecs::Entity>
{
	let (width, height) = kind.get_size();
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::BuildingPlacement {
			width: width,
			height: height,
			valid: vec![false; (width * height) as usize],
			kind: kind,
		},
	));
	Ok(entity)
}

pub fn spawn_blimp(tile_pos: Point2<i32>, world: &mut hecs::World) -> Result<hecs::Entity>
{
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::TilePath { tile_path: vec![] },
		comps::CanMove {
			moving: false,
			flies: true,
		},
		comps::Blimp {
			time_to_leave: 0.,
			state: comps::BlimpState::Spawned,
		},
	));
	Ok(entity)
}

fn get_tile_path(
	from: Point2<i32>, to: Point2<i32>, size: usize, walkable: &[bool], walk_speed: Option<&[f32]>,
	allow_partial: bool,
) -> Vec<Point2<i32>>
{
	let mut ctx = astar::AStarContext::new(size);
	let path = ctx.solve(
		from,
		to,
		|pos| tile_to_idx(pos, size).map_or(true, |idx| !walkable[idx]),
		|pos| tile_to_idx(pos, size).map_or(1., |idx| walk_speed.map_or(1., |ws| 1. / ws[idx])),
	);
	if !path.is_empty() && path[0] != to && !allow_partial
	{
		vec![]
	}
	else
	{
		path
	}
}

#[derive(PartialEq, Eq, Debug)]
enum CursorKind
{
	Normal,
	BuildingPlacement(hecs::Entity),
	Destroy,
}

struct Map
{
	size: usize,
	terrain: Vec<bool>,
	crystal_distance: Vec<i32>,
	world: hecs::World,
	rng: StdRng,
	mouse_pos: Point2<i32>,
	port: Option<hecs::Entity>,
	blimp: Option<hecs::Entity>,
	cursor_kind: CursorKind,
	camera_pos: Vector2<i32>,
	money: i32,
	population: i32,

	want_port: bool,
	want_normal: bool,
	want_destroy: bool,
	want_cafe: bool,
	want_house: bool,
	want_road: bool,
	want_mine: bool,

	show_money_chart: bool,
	money_history: Vec<i32>,
	money_history_len: usize,
	money_mean: f32,
	money_mean_count: i32,

	show_pop_chart: bool,
	pop_history: Vec<i32>,
	pop_history_len: usize,
	pop_mean: f32,
	pop_mean_count: i32,
}

impl Map
{
	fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		let mut rng = StdRng::seed_from_u64(thread_rng().gen::<u16>() as u64);

		state.cache_sprite("data/road.cfg")?;
		state.cache_sprite("data/chart.cfg")?;
		state.cache_sprite("data/cursor.cfg")?;
		state.cache_sprite("data/buttons.cfg")?;
		state.cache_sprite("data/thought_sleepy.cfg")?;
		state.cache_sprite("data/thought_hungry.cfg")?;
		state.cache_sprite("data/plot_1x1.cfg")?;
		state.cache_sprite("data/plot_3x2.cfg")?;
		state.cache_sprite("data/plot_3x3.cfg")?;
		state.cache_sprite("data/terrain.cfg")?;
		state.cache_sprite("data/terrain.cfg")?;
		state.cache_sprite("data/cat1.cfg")?;
		state.cache_sprite("data/cat2.cfg")?;
		state.cache_sprite("data/duck.cfg")?;
		state.cache_sprite("data/crystal1.cfg")?;
		state.cache_sprite("data/crystal2.cfg")?;
		state.cache_sprite("data/house1.cfg")?;
		state.cache_sprite("data/house2.cfg")?;
		state.cache_sprite("data/mine1.cfg")?;
		state.cache_sprite("data/port.cfg")?;
		state.cache_sprite("data/blimp.cfg")?;
		state.cache_sprite("data/building_placement.cfg")?;
		state.cache_sprite("data/cafe.cfg")?;

		let size: i32 = 32;

		let heightmap = diamond_square(size.ilog2() as i32, &mut rng);
		let heightmap = smooth_heightmap(&heightmap);
		let heightmap = lower_heightmap(&heightmap);
		let max = *heightmap.iter().reduce(std::cmp::max).unwrap() as f32;
		let real_size = (heightmap.len() as f32).sqrt() as i32;

		let mut terrain = Vec::with_capacity((size * size) as usize);
		let mut crystal_distance = Vec::with_capacity((size * size) as usize);

		let mut crystal_positions = vec![];
		let mut world = hecs::World::new();
		for y in 0..size
		{
			for x in 0..size
			{
				let val = if x == 0 || x == size - 1 || y == 0 || y == size - 1
				{
					false
				}
				else
				{
					let f = (-((x as f32 - size as f32 / 2.).powf(2.)
						+ (y as f32 - size as f32 / 2.).powf(2.))
						/ 100.)
						.exp();
					(heightmap[(y * real_size + x) as usize] + (max as f32 * 1. * f) as i32)
						> (0.4 * max) as i32
				};
				if val
					&& rng.gen_bool(0.1) && ((x as i32 - size / 2).abs() + (y as i32 - size / 2))
					.abs() > 5
				{
					let pos = Point2::new(x as i32, y as i32);
					spawn_crystal(pos, &mut world, &mut rng)?;
					crystal_positions.push(pos);
				}
				terrain.push(val)
			}
		}

		for y in 0..size
		{
			for x in 0..size
			{
				let mut min_dist = size;
				let tile_pos = Point2::new(x as f32, y as f32);
				for crystal_pos in &crystal_positions
				{
					let dist = (to_f32(*crystal_pos) - tile_pos).magnitude() as i32;
					min_dist = utils::min(dist, min_dist);
				}
				crystal_distance.push(min_dist);
			}
		}

		let from = Point2::new(size / 2, size / 2);
		//terrain[tile_to_idx(from, size).unwrap()] = true;
		spawn_agent(from, &mut world, state, &mut rng)?;
		let from = Point2::new(size / 2 + 1, size / 2);
		spawn_agent(from, &mut world, state, &mut rng)?;

		let start_money = 20000;
		let mut money_history = vec![0; 16];
		let l = money_history.len();
		money_history[l - 1] = start_money;

		let start_pop = 2;
		let mut pop_history = vec![0; 16];
		let l = pop_history.len();
		pop_history[l - 1] = start_pop;

		Ok(Self {
			size: size as usize,
			terrain: terrain,
			world: world,
			rng: rng,
			mouse_pos: Point2::new(
				state.buffer_width as i32 / 2,
				state.buffer_height as i32 / 2,
			),
			port: None,
			blimp: None,
			cursor_kind: CursorKind::Normal,
			camera_pos: Vector2::new(
				size as i32 / 2 * TILE_SIZE - state.buffer_width as i32 / 2,
				size as i32 / 2 * TILE_SIZE - state.buffer_height as i32 / 2,
			),
			money: start_money,
			population: 0,
			want_normal: false,
			want_destroy: false,
			want_cafe: false,
			want_house: false,
			want_road: false,
			want_port: false,
			want_mine: false,
			show_money_chart: false,
			money_mean: start_money as f32,
			money_mean_count: 0,
			money_history_len: 1,
			money_history: money_history,
			show_pop_chart: false,
			pop_mean: start_pop as f32,
			pop_mean_count: 0,
			pop_history_len: 1,
			pop_history: pop_history,
			crystal_distance: crystal_distance,
		})
	}

	fn logic(&mut self, state: &mut game_state::GameState)
		-> Result<Option<game_state::NextScreen>>
	{
		let mut to_die = vec![];

		// History.
		self.money_mean = (self.money_mean * self.money_mean_count as f32 + self.money as f32)
			/ (self.money_mean_count + 1) as f32;
		self.money_mean_count += 1;
		if self.money_mean_count == MEAN_TICKS
		{
			self.money_history.rotate_left(1);
			let l = self.money_history.len();
			self.money_history[l - 1] = (self.money_mean + 0.5) as i32;
			self.money_mean_count = 0;
			self.money_mean = 0.;
			self.money_history_len += 1;
		}

		self.pop_mean = (self.pop_mean * self.pop_mean_count as f32 + self.population as f32)
			/ (self.pop_mean_count + 1) as f32;
		self.pop_mean_count += 1;
		if self.pop_mean_count == MEAN_TICKS
		{
			self.pop_history.rotate_left(1);
			let l = self.pop_history.len();
			self.pop_history[l - 1] = (self.pop_mean + 0.5) as i32;
			self.pop_mean_count = 0;
			self.pop_mean = 0.;
			self.pop_history_len += 1;
		}

		// Input.
		let mut build_kind = None;
		let mut new_cursor = None;
		if state.controls.get_action_state(controls::Action::BuildRoad) > 0.5 || self.want_road
		{
			build_kind = Some(comps::ProviderKind::Road);
		}
		if state
			.controls
			.get_action_state(controls::Action::BuildHouse)
			> 0.5 || self.want_house
		{
			build_kind = Some(comps::ProviderKind::EmptyHouse);
		}
		if (state.controls.get_action_state(controls::Action::BuildPort) > 0.5 || self.want_port)
			&& self.port.is_none()
		{
			build_kind = Some(comps::ProviderKind::Port);
		}
		if state.controls.get_action_state(controls::Action::BuildMine) > 0.5 || self.want_mine
		{
			build_kind = Some(comps::ProviderKind::Mine);
		}
		if state.controls.get_action_state(controls::Action::BuildCafe) > 0.5 || self.want_cafe
		{
			build_kind = Some(comps::ProviderKind::Cafe);
		}
		if self.want_normal
		{
			new_cursor = Some(CursorKind::Normal);
		}
		if state.controls.get_action_state(controls::Action::Destroy) > 0.5 || self.want_destroy
		{
			new_cursor = Some(CursorKind::Destroy);
		}
		if let Some(build_kind) = build_kind
		{
			let building_placement =
				spawn_building_placement(Point2::new(5, 5), build_kind, &mut self.world)?;
			new_cursor = Some(CursorKind::BuildingPlacement(building_placement));
		}

		if let Some(new_cursor) = new_cursor
		{
			match self.cursor_kind
			{
				CursorKind::BuildingPlacement(building_placement) =>
				{
					to_die.push(building_placement);
				}
				_ => (),
			}
			self.cursor_kind = new_cursor;
		}
		self.want_normal = false;
		self.want_destroy = false;
		self.want_cafe = false;
		self.want_house = false;
		self.want_road = false;
		self.want_port = false;
		self.want_mine = false;

		// Camera.
		if self.mouse_pos.x <= 0
		{
			self.camera_pos.x -= state.options.camera_speed;
		}
		if self.mouse_pos.x + 1 >= state.buffer_width as i32
		{
			self.camera_pos.x += state.options.camera_speed;
		}
		if self.mouse_pos.y <= 0
		{
			self.camera_pos.y -= state.options.camera_speed;
		}
		if self.mouse_pos.y + 1 >= state.buffer_height as i32
		{
			self.camera_pos.y += state.options.camera_speed;
		}
		self.camera_pos.x = utils::clamp(
			self.camera_pos.x,
			0,
			self.size as i32 * TILE_SIZE - state.buffer_width as i32,
		);
		self.camera_pos.y = utils::clamp(
			self.camera_pos.y,
			0,
			self.size as i32 * TILE_SIZE - state.buffer_height as i32,
		);

		// Maps.
		let mut buildable = self.terrain.clone();
		let mut walkable = self.terrain.clone();
		let flyable = vec![true; buildable.len()];
		let mut walk_speed = vec![1.; buildable.len()];
		for (_, (position, _)) in self
			.world
			.query::<(&comps::Position, &comps::Crystal)>()
			.iter()
		{
			let tile_pos = pixel_to_tile(position.pos);
			for dy in [0, -1]
			{
				if let Some(idx) = tile_to_idx(Point2::new(tile_pos.x, tile_pos.y + dy), self.size)
				{
					buildable[idx] = false;
					walkable[idx] = false;
				}
			}
		}
		for (_, (position, provider)) in self
			.world
			.query::<(&comps::Position, &comps::Provider)>()
			.iter()
		{
			let tile_pos = pixel_to_tile(position.pos);
			let (width, height) = provider.kind.get_size();
			let start_x = tile_pos.x - width / 2;
			let start_y = tile_pos.y - height + 1;
			for y in 0..height
			{
				for x in 0..width
				{
					let cur_tile_pos = Point2::new(start_x + x, start_y + y);
					if let Some(idx) = tile_to_idx(cur_tile_pos, self.size)
					{
						buildable[idx] = false;
						walkable[idx] = cur_tile_pos == tile_pos;
						walk_speed[idx] = if provider.kind == comps::ProviderKind::Road
						{
							2.
						}
						else
						{
							1.
						};
					}
				}
			}
		}
		for (_, (position, _)) in self
			.world
			.query::<(&comps::Position, &comps::Agent)>()
			.iter()
		{
			let tile_pos = pixel_to_tile(position.pos);
			if let Some(idx) = tile_to_idx(Point2::new(tile_pos.x, tile_pos.y), self.size)
			{
				buildable[idx] = false;
			}
		}

		// BuildingPlacement
		match self.cursor_kind
		{
			CursorKind::BuildingPlacement(building_placement) =>
			{
				let (position, building_placement) = self
					.world
					.query_one_mut::<(&mut comps::Position, &mut comps::BuildingPlacement)>(
						building_placement,
					)
					.unwrap();
				let tile_pos = pixel_to_tile(
					self.mouse_pos + self.camera_pos + Vector2::new(TILE_SIZE, TILE_SIZE) / 2,
				);
				let start_x = tile_pos.x - building_placement.width / 2;
				let start_y = tile_pos.y - building_placement.height + 1;
				for y in 0..building_placement.height
				{
					for x in 0..building_placement.width
					{
						let free = tile_to_idx(Point2::new(x + start_x, y + start_y), self.size)
							.map_or(false, |idx| buildable[idx]);
						building_placement.valid[(x + y * building_placement.width) as usize] =
							free;
					}
				}
				position.pos = tile_to_pixel(tile_pos);
			}
			_ => (),
		}

		// TilePath
		let mut stuck = vec![];
		for (id, (position, can_move, tile_path)) in self
			.world
			.query::<(
				&mut comps::Position,
				&mut comps::CanMove,
				&mut comps::TilePath,
			)>()
			.iter()
		{
			if tile_path.tile_path.is_empty()
			{
				can_move.moving = false;
				continue;
			}
			can_move.moving = true;

			let target_tile = *tile_path.tile_path.last().unwrap();
			let target_idx = tile_to_idx(target_tile, self.size).unwrap();
			if !walkable[target_idx] && !can_move.flies
			{
				can_move.moving = false;
				tile_path.tile_path.clear();
				stuck.push(id);
				continue;
			}
			let target = tile_to_pixel(target_tile);
			let speed = if can_move.flies
			{
				2
			}
			else
			{
				walk_speed[target_idx] as i32
			};
			if target.x > position.pos.x
			{
				position.pos.x += speed;
				position.dir = 0;
			}
			if target.y > position.pos.y
			{
				position.pos.y += speed;
				position.dir = 1;
			}
			if target.x < position.pos.x
			{
				position.pos.x -= speed;
				position.dir = 2;
			}
			if target.y < position.pos.y
			{
				position.pos.y -= speed;
				position.dir = 3;
			}
			if target == position.pos
			{
				tile_path.tile_path.pop().unwrap();
			}
		}
		for id in stuck
		{
			if let Ok(agent) = self.world.query_one_mut::<&mut comps::Agent>(id)
			{
				println!("Stuck: {:?}", id);
				agent.cur_provider = None;
			}
		}

		// Providers.
		let mut indicators = vec![];
		let mut kind_to_providers = HashMap::new();
		let mut provider_to_slots = HashMap::new();
		for (id, (position, provider)) in self
			.world
			.query::<(&comps::Position, &mut comps::Provider)>()
			.iter()
		{
			if provider.kind == comps::ProviderKind::Road
			{
				continue;
			}
			if state.time() > provider.time_to_maintain
			{
				let maintenance_cost = provider.kind.get_cost() / 10;
				if provider.kind.get_cost() != 0
				{
					let (text, color) = get_money_indicator(-maintenance_cost, true);
					indicators.push((
						pixel_to_tile(position.pos) - Vector2::new(0, 1),
						text,
						color,
					));
				}
				self.money -= maintenance_cost;
				provider.time_to_maintain = state.time() + 10.;
			}
			if provider.num_occupants == provider.max_occupants
			{
				continue;
			}
			provider_to_slots.insert(id, provider.max_occupants - provider.num_occupants);
			let providers = kind_to_providers
				.entry(provider.kind)
				.or_insert_with(Vec::new);
			providers.push((id, pixel_to_tile(position.pos)));
		}

		// Port
		let mut port_pos = None;
		if let Some(port_id) = self.port
		{
			if !self.world.contains(port_id)
			{
				self.port = None;
			}
			else
			{
				let port_position = self
					.world
					.query_one_mut::<&comps::Position>(port_id)
					.unwrap()
					.clone();
				port_pos = Some(pixel_to_tile(port_position.pos));
				if self.blimp.is_none()
				{
					let warp_location = get_warp_location(&mut self.rng, self.size);
					self.blimp = Some(spawn_blimp(
						Point2::new(warp_location.0, warp_location.1),
						&mut self.world,
					)?);
				}
			}
		}

		// Agent.
		self.population = 0;
		let mut providers_to_change = vec![];
		let mut providers_to_work = vec![];
		let mut providers_to_house_claim = vec![];
		for (agent_id, (position, tile_path, can_move, agent, agent_draw)) in self
			.world
			.query::<(
				&comps::Position,
				&mut comps::TilePath,
				&mut comps::CanMove,
				&mut comps::Agent,
				&mut comps::AgentDraw,
			)>()
			.iter()
		{
			self.population += 1;
			agent.hunger += 1;
			agent.sleepyness += 1;
			agent.hunger = utils::clamp(agent.hunger, 0, MAX_HUNGER);
			agent.sleepyness = utils::clamp(agent.sleepyness, 0, MAX_SLEEPYNESS);
			if let Some(provider) = agent.cur_provider
			{
				if !self.world.contains(provider)
				{
					agent.cur_provider = None;
				}
			}
			if let Some(house) = agent.house
			{
				if !self.world.contains(house)
				{
					agent.house = None;
				}
			}
			if agent.leaving && self.port.is_none()
			{
				agent.leaving = false;
			}
			if can_move.moving || agent.leaving
			{
				continue;
			}
			if state.time() > agent.time_to_work
			{
				if let Some(provider_id) = agent.cur_provider
				{
					providers_to_work.push((provider_id, agent_id));
				}
				agent.time_to_work = state.time() + 1.;
			}
			if state.time() < agent.time_to_think
			{
				continue;
			}
			agent.time_to_think = state.time() + 7. + self.rng.gen::<f64>();
			agent_draw.thought = None;

			if let Some(provider) = agent.cur_provider.take()
			{
				providers_to_change.push((provider, -1));
			}

			let mut actions = vec![
				(None, 1.0),
				(
					Some(comps::ProviderKind::Cafe),
					5. * agent.hunger as f32 / MAX_HUNGER as f32,
				),
			];
			if agent.house.is_some()
			{
				actions.push((
					Some(comps::ProviderKind::TakenHouse(agent_id)),
					5. * agent.sleepyness as f32 / MAX_SLEEPYNESS as f32,
				));
			}
			else
			{
				actions.push((
					Some(comps::ProviderKind::EmptyHouse),
					5. * agent.sleepyness as f32 / MAX_SLEEPYNESS as f32,
				));
			}

			for (kind, weight) in [
				(comps::ProviderKind::Mine, 2.0),
				(
					comps::ProviderKind::Port,
					5. * utils::max(
						agent.hunger as f32 / MAX_HUNGER as f32,
						agent.sleepyness as f32 / MAX_SLEEPYNESS as f32,
					)
					.powf(4.),
				),
				(comps::ProviderKind::Plot1x1, 3.0),
				(comps::ProviderKind::Plot3x2, 3.0),
				(comps::ProviderKind::Plot3x3, 3.0),
			]
			{
				if kind_to_providers.contains_key(&kind)
				{
					actions.push((Some(kind), weight));
				}
			}

			let (action, _) = actions
				.choose_weighted(&mut self.rng, |(_, weight)| *weight)
				.unwrap();
			println!("{agent_id:?} {actions:?}");
			println!(
				"{agent_id:?}: {action:?} hunger: {} sleepyness: {}",
				agent.hunger, agent.sleepyness
			);

			if let Some(kind) = action
			{
				if let Some(providers) = kind_to_providers.get(kind)
				{
					let mut provider_and_scores = Vec::with_capacity(providers.len());
					let mut total_score = 0.;
					for (provider_id, provider_pos) in providers
					{
						let dist = (to_f32(*provider_pos) - to_f32(position.pos)).magnitude();
						let score = 1. / (1. + dist);
						total_score += score;
						if provider_to_slots[provider_id] <= 0
						{
							continue;
						}
						provider_and_scores.push(((*provider_id, *provider_pos), score));
					}
					for provider_and_score in &mut provider_and_scores
					{
						provider_and_score.1 += 0.1 * total_score * self.rng.gen::<f32>();
					}
					// Descending sort.
					provider_and_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
					for ((provider_id, provider_pos), _) in &provider_and_scores
					{
						let cand_tile_path = get_tile_path(
							pixel_to_tile(position.pos),
							*provider_pos,
							self.size,
							&walkable,
							Some(&walk_speed),
							false,
						);
						if !cand_tile_path.is_empty()
						{
							tile_path.tile_path = cand_tile_path;
							can_move.moving = true;
							providers_to_change.push((*provider_id, 1));
							*provider_to_slots.get_mut(&provider_id).unwrap() -= 1;
							agent.cur_provider = Some(*provider_id);
							match *kind
							{
								comps::ProviderKind::EmptyHouse =>
								{
									agent.house = Some(*provider_id);
									providers_to_house_claim.push((*provider_id, agent_id));
								}
								comps::ProviderKind::Port =>
								{
									agent.leaving = true;
								}
								_ => (),
							}
							break;
						}
					}
				}
				if !can_move.moving
				{
					match kind
					{
						comps::ProviderKind::EmptyHouse | comps::ProviderKind::TakenHouse(_) =>
						{
							agent_draw.thought = Some("data/thought_sleepy.cfg".to_string())
						}
						comps::ProviderKind::Cafe =>
						{
							agent_draw.thought = Some("data/thought_hungry.cfg".to_string())
						}
						_ => (),
					}
					println!("Failed to act!");
				}
			}
			else
			{
				let cur_pos = pixel_to_tile(position.pos);
				let mut target =
					cur_pos + Vector2::new(self.rng.gen_range(-3..=3), self.rng.gen_range(-3..=3));
				target.x = utils::clamp(target.x, 0, self.size as i32 - 1);
				target.y = utils::clamp(target.y, 0, self.size as i32 - 1);
				let cand_tile_path = get_tile_path(
					cur_pos,
					target,
					self.size,
					&walkable,
					Some(&walk_speed),
					true,
				);
				tile_path.tile_path = cand_tile_path;
				can_move.moving = true;
			}
		}
		for (provider_id, agent_id) in providers_to_house_claim
		{
			let (provider, scenery_draw) = self
				.world
				.query_one_mut::<(&mut comps::Provider, &mut comps::SceneryDraw)>(provider_id)
				.unwrap();
			provider.kind = comps::ProviderKind::TakenHouse(agent_id);
			scenery_draw.variant = provider.kind.get_sprite(&mut self.rng).1
		}
		for (provider_id, agent_id) in providers_to_work
		{
			let mut plot_complete = None;
			if let Ok((position, plot)) = self
				.world
				.query_one_mut::<(&comps::Position, &mut comps::Plot)>(provider_id)
			{
				plot.work_done += 1;
				indicators.push((
					pixel_to_tile(position.pos) - Vector2::new(0, 1),
					format!("{}/{}", plot.work_done, plot.work_total),
					Color::from_rgb_f(0.9, 0.9, 0.3),
				));
				if plot.work_done >= plot.work_total
				{
					plot_complete = Some(plot.kind);
				}
			}

			let mut hunger_change = 0;
			let mut sleepyness_change = 0;

			{
				let (position, provider) = self
					.world
					.query_one_mut::<(&comps::Position, &mut comps::Provider)>(provider_id)
					.unwrap();
				let mine_money = 500
					/ self.crystal_distance
						[tile_to_idx(pixel_to_tile(position.pos), self.size).unwrap()];
				match provider.kind
				{
					comps::ProviderKind::TakenHouse(_) =>
					{
						sleepyness_change = -500;
						indicators.push((
							pixel_to_tile(position.pos) - Vector2::new(0, 1),
							"zZz".to_string(),
							Color::from_rgb_f(0.4, 0.3, 0.9),
						));
					}
					comps::ProviderKind::Cafe =>
					{
						if self.money < 50
						{
							indicators.push((
								pixel_to_tile(position.pos) - Vector2::new(0, 1),
								"No Money!".to_string(),
								Color::from_rgb_f(0.8, 0.2, 0.3),
							));
						}
						else
						{
							hunger_change = -500;
							self.money -= 50;
							let (text, color) = get_money_indicator(-50, true);
							indicators.push((
								pixel_to_tile(position.pos) - Vector2::new(0, 1),
								text,
								color,
							));
						}
					}
					comps::ProviderKind::Mine =>
					{
						self.money += mine_money;
						let (text, color) = get_money_indicator(mine_money, true);
						indicators.push((
							pixel_to_tile(position.pos) - Vector2::new(0, 1),
							text,
							color,
						));
						hunger_change = 50;
					}
					_ => hunger_change = 50,
				}
			}
			{
				let agent = self
					.world
					.query_one_mut::<&mut comps::Agent>(agent_id)
					.unwrap();
				agent.hunger += hunger_change;
				agent.sleepyness += sleepyness_change;
				if plot_complete.is_some()
				{
					//println!("Finished");
					agent.cur_provider = None;
				}
			}
			if let Some(kind) = plot_complete
			{
				let (provider, scenery_draw) = self
					.world
					.query_one_mut::<(&mut comps::Provider, &mut comps::SceneryDraw)>(provider_id)
					.unwrap();
				provider.kind = kind;
				provider.num_occupants = 0;
				if kind == comps::ProviderKind::Port
				{
					self.port = Some(provider_id);
				}
				let (sprite, variant) = kind.get_sprite(&mut self.rng);
				scenery_draw.sprite = sprite.to_string();
				scenery_draw.variant = variant;
				self.world.remove_one::<comps::Plot>(provider_id)?;
			}

			//println!("Work unit!");
		}
		for (provider_id, change) in providers_to_change
		{
			let mut provider = self.world.get::<&mut comps::Provider>(provider_id).unwrap();
			provider.num_occupants += change;
		}
		for (tile_pos, text, color) in indicators
		{
			spawn_indicator(tile_pos, text, color, &mut self.world, state)?;
		}

		// Agent -> AgentDraw correspondence
		for (_, (agent, agent_draw, can_move)) in self
			.world
			.query::<(&comps::Agent, &mut comps::AgentDraw, &comps::CanMove)>()
			.iter()
		{
			agent_draw.visible = can_move.moving || agent.cur_provider.is_none();
		}

		// Indicator
		for (id, (position, indicator)) in self
			.world
			.query::<(&mut comps::Position, &comps::Indicator)>()
			.iter()
		{
			if state.time() > indicator.time_to_die
			{
				to_die.push(id);
				continue;
			}
			position.pos.y -= 1;
		}

		// Blimp
		let mut collect_agents = false;
		let mut add_agents = false;
		if let Some(blimp_id) = self.blimp
		{
			let warp_location = get_warp_location(&mut self.rng, self.size);
			let (blimp_pos, blimp, can_move, tile_path) = self
				.world
				.query_one_mut::<(
					&comps::Position,
					&mut comps::Blimp,
					&mut comps::CanMove,
					&mut comps::TilePath,
				)>(blimp_id)
				.unwrap();
			match blimp.state
			{
				comps::BlimpState::Spawned =>
				{
					if let Some(port_pos) = port_pos
					{
						tile_path.tile_path = get_tile_path(
							pixel_to_tile(blimp_pos.pos),
							port_pos,
							self.size,
							&flyable,
							None,
							false,
						);
						blimp.state = comps::BlimpState::Arriving;
					}
					else
					{
						blimp.state = comps::BlimpState::Leaving;
					}
				}
				comps::BlimpState::Arriving =>
				{
					if !can_move.moving
					{
						add_agents = true;
						blimp.time_to_leave = state.time() + 5.;
						blimp.state = comps::BlimpState::Waiting;
					}
				}
				comps::BlimpState::Waiting =>
				{
					if state.time() > blimp.time_to_leave
					{
						tile_path.tile_path = get_tile_path(
							pixel_to_tile(blimp_pos.pos),
							Point2::new(warp_location.0, warp_location.1),
							self.size,
							&flyable,
							None,
							false,
						);
						blimp.state = comps::BlimpState::Leaving;
						collect_agents = true;
					}
				}
				comps::BlimpState::Leaving =>
				{
					if !can_move.moving
					{
						self.blimp = None;
						to_die.push(blimp_id);
					}
				}
			}
		}
		if add_agents
		{
			if let Some(port_pos) = port_pos
			{
				let mut num_to_spawn = self.rng.gen_range(0..3);
				if self.population > 50
				{
					num_to_spawn = 0;
				}
                if self.population == 0
                {
                    num_to_spawn = 2;
                }
				for _ in 0..num_to_spawn
				{
					spawn_agent(port_pos, &mut self.world, state, &mut self.rng)?;
				}
			}
		}
		let mut houses_to_free = vec![];
		let mut num_left = 0;
		if collect_agents
		{
			for (agent_id, (can_move, agent)) in self
				.world
				.query::<(&comps::CanMove, &comps::Agent)>()
				.iter()
			{
				if !can_move.moving && agent.leaving
				{
					if let Some(house_id) = agent.house
					{
						houses_to_free.push(house_id);
					}
					to_die.push(agent_id);
					num_left += 1;
				}
			}
		}
		if let Some(port_id) = self.port
		{
			let provider = self
				.world
				.query_one_mut::<&mut comps::Provider>(port_id)
				.unwrap();
			provider.num_occupants -= num_left;
		}
		for house_id in houses_to_free
		{
			let (provider, scenery_draw) = self
				.world
				.query_one_mut::<(&mut comps::Provider, &mut comps::SceneryDraw)>(house_id)
				.unwrap();
			provider.kind = comps::ProviderKind::EmptyHouse;
			scenery_draw.variant = provider.kind.get_sprite(&mut self.rng).1;
		}

		// Remove dead entities
		to_die.sort();
		to_die.dedup();
		for id in to_die
		{
			//println!("died {id:?}");
			self.world.despawn(id)?;
		}

		Ok(None)
	}

	fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
		state.controls.decode_event(event);
		match *event
		{
			Event::MouseAxes { x, y, .. } =>
			{
				let (x, y) = state.transform_mouse(x as f32, y as f32);
				self.mouse_pos = Point2::new(x as i32, y as i32);
			}
			Event::KeyDown {
				keycode: KeyCode::Escape,
				..
			} =>
			{
				self.want_normal = true;
			}
			Event::MouseButtonDown { button, .. } =>
			{
				let mut indicators = vec![];
				if button == 1
				{
					let offt = 16 * NUM_BUTTONS / 2;
					let left_x = state.buffer_width as i32 / 2 - offt;
					let left_y = state.buffer_height as i32 - 16;
					let right_x = state.buffer_width as i32 / 2 + offt;
					let right_y = state.buffer_height as i32;

					if self.mouse_pos.x >= left_x
						&& self.mouse_pos.x < right_x
						&& self.mouse_pos.y >= left_y
						&& self.mouse_pos.y < right_y
					{
						let button = (self.mouse_pos.x - left_x) / 16;
						match button
						{
							0 => self.want_normal = true,
							1 => self.want_destroy = true,
							2 => self.want_house = true,
							3 => self.want_mine = true,
							4 => self.want_port = true,
							5 => self.want_cafe = true,
							6 => self.want_road = true,
							7 =>
							{
								self.show_money_chart = !self.show_money_chart;
								self.show_pop_chart = false;
							}
							8 =>
							{
								self.show_pop_chart = !self.show_pop_chart;
								self.show_money_chart = false;
							}
							_ => (),
						}
					}
					else
					{
						let mut spawn = None;
						match self.cursor_kind
						{
							CursorKind::BuildingPlacement(building_placement) =>
							{
								let (position, building_placement) = self
									.world
									.query_one_mut::<(&mut comps::Position, &mut comps::BuildingPlacement)>(
										building_placement,
									)
									.unwrap();
								if building_placement.valid.iter().all(|x| *x)
								{
									spawn = Some((
										pixel_to_tile(position.pos),
										building_placement.kind,
									));
								}
							}
							CursorKind::Destroy =>
							{
								let tile_pos = pixel_to_tile(
									self.mouse_pos
										+ self.camera_pos + Vector2::new(TILE_SIZE, TILE_SIZE) / 2,
								);
								let mut to_die = None;
								for (id, (position, provider)) in self
									.world
									.query::<(&comps::Position, &comps::Provider)>()
									.iter()
								{
									let provider_tile_pos = pixel_to_tile(position.pos);
									let (width, height) = provider.kind.get_size();
									let start_x = provider_tile_pos.x - width / 2;
									let start_y = provider_tile_pos.y - height + 1;
									if tile_pos.x >= start_x
										&& tile_pos.x < start_x + width && tile_pos.y >= start_y
										&& tile_pos.y < start_y + height
									{
										to_die = Some(id);
										self.money += provider.kind.get_cost();
										break;
									}
								}
								if let Some(id) = to_die
								{
									let position =
										self.world.query_one_mut::<&comps::Position>(id).unwrap();
									let position = position.clone();
									let mut cost = 0;
									if let Ok(plot) = self.world.query_one_mut::<&comps::Plot>(id)
									{
										cost = plot.kind.get_cost();
									}
									self.money += cost;
									let (text, color) = get_money_indicator(cost, true);
									indicators.push((pixel_to_tile(position.pos), text, color));
									self.world.despawn(id)?;
								}
							}
							_ => (),
						}
						if let Some((position, kind)) = spawn
						{
							if self.money >= kind.get_cost()
							{
								spawn_provider(
									position,
									kind,
									&mut self.world,
									state,
									&mut self.rng,
								)?;
								let (text, color) = get_money_indicator(-kind.get_cost(), true);
								indicators.push((position, text, color));
								self.money -= kind.get_cost();
							}
							else
							{
								indicators.push((
									position,
									"No Money!".to_string(),
									Color::from_rgb_f(0.8, 0.2, 0.3),
								));
							}
						}
					}
					for (tile_pos, text, color) in indicators
					{
						spawn_indicator(tile_pos, text, color, &mut self.world, state)?;
					}
				}
			}
			_ => (),
		}
		Ok(None)
	}

	fn draw(&mut self, state: &game_state::GameState) -> Result<()>
	{
		state
			.core
			.clear_to_color(Color::from_rgb_f(0.05, 0.05, 0.1));
		let terrain_sprite = state.get_sprite("data/terrain.cfg").unwrap();

		// Terrain
		let mut rng = StdRng::seed_from_u64(0);
		state.core.hold_bitmap_drawing(true);
		for y in 0..self.size as i32 - 1
		{
			for x in 0..self.size as i32 - 1
			{
				let t1 = self.terrain[tile_to_idx(Point2::new(x, y), self.size).unwrap()] as i32;
				let t2 =
					self.terrain[tile_to_idx(Point2::new(x + 1, y), self.size).unwrap()] as i32;
				let t3 =
					self.terrain[tile_to_idx(Point2::new(x, y + 1), self.size).unwrap()] as i32;
				let t4 =
					self.terrain[tile_to_idx(Point2::new(x + 1, y + 1), self.size).unwrap()] as i32;

				let tile = decode_tile([t1, t2, t3, t4], &mut rng);
				terrain_sprite.draw(
					to_f32(tile_to_pixel(Point2::new(x, y)) - self.camera_pos),
					tile,
					Color::from_rgb_f(1., 1., 1.),
					state,
				);
			}
		}

		// Scenery
		for (_, (position, scenery_draw)) in self
			.world
			.query::<(&comps::Position, &comps::SceneryDraw)>()
			.iter()
		{
			let sprite = state.get_sprite(&scenery_draw.sprite).unwrap();

			sprite.draw(
				to_f32(position.pos - self.camera_pos),
				scenery_draw.variant,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}

		// Agents
		for (_, (position, can_move, agent_draw)) in self
			.world
			.query::<(&comps::Position, &comps::CanMove, &comps::AgentDraw)>()
			.iter()
		{
			if !agent_draw.visible
			{
				continue;
			}
			let sprite = state.get_sprite(&agent_draw.sprite).unwrap();

			let variant = if can_move.moving
			{
				let offt = [0, 4, 0, 8][(state.tick / 5) as usize % 4];
				position.dir + offt
			}
			else
			{
				position.dir
			};

			sprite.draw(
				to_f32(position.pos - self.camera_pos),
				variant,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);

			if let Some(thought) = agent_draw.thought.as_ref()
			{
				let sprite = state.get_sprite(&thought).unwrap();
				sprite.draw(
					to_f32(position.pos - self.camera_pos),
					0,
					Color::from_rgb_f(1., 1., 1.),
					state,
				);
			}
		}

		// Blimp
		if let Some(blimp) = self.blimp
		{
			let position = self.world.query_one_mut::<&comps::Position>(blimp).unwrap();
			let sprite = state.get_sprite("data/blimp.cfg").unwrap();
			sprite.draw(
				to_f32(position.pos - self.camera_pos),
				0,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}

		// Indicator
		for (_, (position, indicator)) in self
			.world
			.query::<(&comps::Position, &comps::Indicator)>()
			.iter()
		{
			let text_pos = position.pos - self.camera_pos;
			state.core.draw_text(
				&state.ui_font,
				indicator.color,
				text_pos.x as f32,
				text_pos.y as f32,
				FontAlign::Centre,
				&indicator.text,
			);
		}

		// Cursor
		match self.cursor_kind
		{
			CursorKind::BuildingPlacement(building_placement) =>
			{
				let (position, building_placement) = self
					.world
					.query_one_mut::<(&comps::Position, &comps::BuildingPlacement)>(
						building_placement,
					)
					.unwrap();
				let sprite = state.get_sprite("data/building_placement.cfg").unwrap();
				let tile_pos = pixel_to_tile(position.pos);
				let start_x = tile_pos.x - building_placement.width / 2;
				let start_y = tile_pos.y - building_placement.height + 1;
				let cost = building_placement.kind.get_cost();
				for y in 0..building_placement.height
				{
					for x in 0..building_placement.width
					{
						let pos = tile_to_pixel(Point2::new(start_x + x, start_y + y));
						sprite.draw(
							to_f32(pos - self.camera_pos),
							(!building_placement.valid[(x + y * building_placement.width) as usize])
								as i32,
							Color::from_rgb_f(1., 1., 1.),
							state,
						);
					}
				}
				let text_pos = to_f32(
					tile_to_pixel(Point2::new(start_x, start_y))
						- self.camera_pos - Vector2::new(TILE_SIZE / 2, TILE_SIZE),
				);
				state.core.draw_text(
					&state.ui_font,
					Color::from_rgb(208, 248, 171),
					text_pos.x,
					text_pos.y,
					FontAlign::Left,
					&format!("${}", cost),
				);
			}
			_ => (),
		}
		state.core.hold_bitmap_drawing(false);

		let (text, color) = get_money_indicator(self.money, false);
		state
			.core
			.draw_text(&state.ui_font, color, 0., 0., FontAlign::Left, &text);
		state.core.draw_text(
			&state.ui_font,
			Color::from_rgb_f(0.9, 0.9, 0.3),
			state.buffer_width - 56.,
			0.,
			FontAlign::Left,
			&format!("Pop: {}", self.population),
		);

		if self.show_money_chart || self.show_pop_chart
		{
			let sprite = state.get_sprite("data/chart.cfg").unwrap();
			sprite.draw(
				Point2::new(
					state.buffer_width / 2. - 64.,
					state.buffer_height / 2. - 64.,
				),
				0,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);

			let (history, history_len) = if self.show_money_chart
			{
				(&self.money_history, self.money_history_len)
			}
			else
			{
				(&self.pop_history, self.pop_history_len)
			};

			let mut max = *history.iter().reduce(std::cmp::max).unwrap() as f32;
			let min = *history.iter().reduce(std::cmp::min).unwrap() as f32;
			if max == min
			{
				max = min + 1.;
			}

			for i in history.len() - utils::min(history.len(), history_len)..history.len() - 1
			{
				let y1 = history[i] as f32;
				let y2 = history[i + 1] as f32;

				let h = 48.;
				let w = 64.;
				let y1 = h - (y1 - min) / (max - min) * h;
				let y2 = h - (y2 - min) / (max - min) * h;
				let x1 = i as f32 / history.len() as f32 * w;
				let x2 = (i + 1) as f32 / history.len() as f32 * w;

				let offt_x = state.buffer_width / 2. - w / 2.;
				let offt_y = state.buffer_height / 2. - h / 2.;
				state.prim.draw_line(
					x1 + offt_x,
					y1 + offt_y,
					x2 + offt_x,
					y2 + offt_y,
					Color::from_rgb_f(0.8, 0.2, 0.3),
					-1.,
				);
			}
			let text = if self.show_money_chart
			{
				get_money_indicator(max as i32, false).0
			}
			else
			{
				format!("{} pop", max as i32)
			};
			state.core.draw_text(
				&state.ui_font,
				Color::from_rgb_f(0., 0., 0.),
				state.buffer_width / 2. - 64. + 8.,
				state.buffer_height / 2. - 48. + 8.,
				FontAlign::Left,
				&text,
			);

			let text = if self.show_money_chart
			{
				get_money_indicator(min as i32, false).0
			}
			else
			{
				format!("{} pop", min as i32)
			};
			state.core.draw_text(
				&state.ui_font,
				Color::from_rgb_f(0., 0., 0.),
				state.buffer_width / 2. - 64. + 8.,
				state.buffer_height / 2. + 48. - 16.,
				FontAlign::Left,
				&text,
			);
		}

		let sprite = state.get_sprite("data/buttons.cfg").unwrap();
		let offt = 16. * NUM_BUTTONS as f32 / 2.;
		for i in 0..NUM_BUTTONS
		{
			sprite.draw(
				Point2::new(
					state.buffer_width / 2. - offt + i as f32 * 16.,
					state.buffer_height - 16.,
				),
				i,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}

		let sprite = state.get_sprite("data/cursor.cfg").unwrap();
		let variant = match self.cursor_kind
		{
			CursorKind::Normal | CursorKind::BuildingPlacement(_) => 0,
			CursorKind::Destroy => 1,
		};
		sprite.draw(
			to_f32(self.mouse_pos),
			variant,
			Color::from_rgb_f(1., 1., 1.),
			state,
		);

		Ok(())
	}
}
