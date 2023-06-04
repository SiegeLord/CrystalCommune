use crate::error::Result;
use crate::{astar, components as comps, controls, game_state, sprite, utils};
use allegro::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

use std::collections::HashMap;

static TILE_SIZE: i32 = 16;
static MAX_SLEEPYNESS: i32 = 3000;

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

pub fn spawn_agent<T: Rng>(
	tile_pos: Point2<i32>, world: &mut hecs::World, _state: &mut game_state::GameState,
	_rng: &mut T,
) -> Result<hecs::Entity>
{
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::TilePath { tile_path: vec![] },
		comps::AgentDraw {
			sprite: "data/cat1.cfg".to_string(),
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

pub fn spawn_provider(
	tile_pos: Point2<i32>, mut kind: comps::ProviderKind, world: &mut hecs::World,
	_state: &mut game_state::GameState,
) -> Result<hecs::Entity>
{
	let size = kind.get_size();
	let real_kind = kind;
	match size
	{
		(3, 2) => kind = comps::ProviderKind::Plot3x2,
		(3, 3) => kind = comps::ProviderKind::Plot3x3,
		_ => panic!("Unknown plot size: {:?}", size),
	}

	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::SceneryDraw {
			sprite: kind.get_sprite().to_string(),
		},
		comps::Provider {
			kind: kind,
			num_occupants: 0,
			max_occupants: kind.get_max_occupants(),
		},
		comps::Plot {
			kind: real_kind,
			work_left: 2,
		},
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
	from: Point2<i32>, to: Point2<i32>, size: usize, walkable: &[bool], allow_partial: bool,
) -> Vec<Point2<i32>>
{
	let mut ctx = astar::AStarContext::new(size);
	let path = ctx.solve(
		from,
		to,
		|pos| tile_to_idx(pos, size).map_or(true, |idx| !walkable[idx]),
		|_| 1.,
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
	world: hecs::World,
	rng: StdRng,
	mouse_pos: Point2<i32>,
	port: Option<hecs::Entity>,
	blimp: Option<hecs::Entity>,
	cursor_kind: CursorKind,
	camera_pos: Vector2<i32>,
}

impl Map
{
	fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		let size = 16;
		let mut terrain = Vec::with_capacity(size * size);
		let mut rng = StdRng::seed_from_u64(0);

		state.cache_sprite("data/thought_sleepy.cfg")?;
		state.cache_sprite("data/plot_3x2.cfg")?;
		state.cache_sprite("data/plot_3x3.cfg")?;
		state.cache_sprite("data/terrain.cfg")?;
		state.cache_sprite("data/terrain.cfg")?;
		state.cache_sprite("data/cat1.cfg")?;
		state.cache_sprite("data/crystal1.cfg")?;
		state.cache_sprite("data/crystal2.cfg")?;
		state.cache_sprite("data/empty_house1.cfg")?;
		state.cache_sprite("data/house1.cfg")?;
		state.cache_sprite("data/office1.cfg")?;
		state.cache_sprite("data/port.cfg")?;
		state.cache_sprite("data/blimp.cfg")?;
		state.cache_sprite("data/building_placement.cfg")?;

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
					//true
					rng.gen_bool(0.9)
				};
				terrain.push(val)
			}
		}
		let mut world = hecs::World::new();
		let from = Point2::new(1, 1);
		//terrain[tile_to_idx(from, size).unwrap()] = true;
		spawn_agent(from, &mut world, state, &mut rng)?;
		let from = Point2::new(9, 1);
		spawn_agent(from, &mut world, state, &mut rng)?;

		//spawn_provider(
		//	Point2::new(7, 7),
		//	comps::ProviderKind::EmptyHouse,
		//	&mut world,
		//	state,
		//)?;
		//spawn_provider(
		//	Point2::new(6, 2),
		//	comps::ProviderKind::EmptyHouse,
		//	&mut world,
		//	state,
		//)?;
		//spawn_provider(
		//	Point2::new(2, 6),
		//	comps::ProviderKind::Office,
		//	&mut world,
		//	state,
		//)?;
		//spawn_provider(
		//	Point2::new(4, 5),
		//	comps::ProviderKind::Office,
		//	&mut world,
		//	state,
		//)?;

		Ok(Self {
			size: size,
			terrain: terrain,
			world: world,
			rng: rng,
			mouse_pos: Point2::new(0, 0),
			port: None,
			blimp: None,
			cursor_kind: CursorKind::Normal,
			camera_pos: Vector2::new(0, 0),
		})
	}

	fn logic(&mut self, state: &mut game_state::GameState)
		-> Result<Option<game_state::NextScreen>>
	{
		let mut to_die = vec![];

		// Input.
		let mut build_kind = None;
		let mut new_cursor = None;
		if state
			.controls
			.get_action_state(controls::Action::BuildHouse)
			> 0.5
		{
			build_kind = Some(comps::ProviderKind::EmptyHouse);
		}
		if state.controls.get_action_state(controls::Action::BuildPort) > 0.5 && self.port.is_none()
		{
			build_kind = Some(comps::ProviderKind::Port);
		}
		if state
			.controls
			.get_action_state(controls::Action::BuildOffice)
			> 0.5
		{
			build_kind = Some(comps::ProviderKind::Office);
		}
		if state.controls.get_action_state(controls::Action::Destroy) > 0.5
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
			if !walkable[tile_to_idx(target_tile, self.size).unwrap()] && !can_move.flies
			{
				can_move.moving = false;
				tile_path.tile_path.clear();
				stuck.push(id);
				continue;
			}
			let target = tile_to_pixel(target_tile);
			if target.x > position.pos.x
			{
				position.pos.x += 1;
				position.dir = 0;
			}
			if target.y > position.pos.y
			{
				position.pos.y += 1;
				position.dir = 1;
			}
			if target.x < position.pos.x
			{
				position.pos.x -= 1;
				position.dir = 2;
			}
			if target.y < position.pos.y
			{
				position.pos.y -= 1;
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

		// Index the providers.
		let mut kind_to_providers = HashMap::new();
		let mut provider_to_slots = HashMap::new();
		for (id, (position, provider)) in self
			.world
			.query::<(&comps::Position, &comps::Provider)>()
			.iter()
		{
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
			agent.hunger += 1;
			agent.sleepyness += 1;
			agent.hunger = utils::clamp(agent.hunger, 0, MAX_SLEEPYNESS);
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
			agent.time_to_think = state.time() + 5. + self.rng.gen::<f64>();
			agent_draw.thought = None;

			if let Some(provider) = agent.cur_provider.take()
			{
				providers_to_change.push((provider, -1));
			}

			let mut actions = vec![(None, 1.0)];
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
				(comps::ProviderKind::Office, 1.0),
				(
					comps::ProviderKind::Port,
					0.2 * agent.sleepyness as f32 / MAX_SLEEPYNESS as f32,
				),
				(comps::ProviderKind::Plot3x2, 2.0),
				(comps::ProviderKind::Plot3x3, 2.0),
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
					if kind.is_house()
					{
						agent_draw.thought = Some("data/thought_sleepy.cfg".to_string());
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
				let cand_tile_path = get_tile_path(cur_pos, target, self.size, &walkable, true);
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
			scenery_draw.sprite = provider.kind.get_sprite().to_string();
		}
		for (provider_id, agent_id) in providers_to_work
		{
			let mut plot_complete = None;
			if let Ok(plot) = self.world.query_one_mut::<&mut comps::Plot>(provider_id)
			{
				plot.work_left -= 1;
				if plot.work_left <= 0
				{
					plot_complete = Some(plot.kind);
				}
			}
			let mut hunger_change = 0;
			let mut sleepyness_change = 0;
			{
				let provider = self
					.world
					.query_one_mut::<&mut comps::Provider>(provider_id)
					.unwrap();
				match provider.kind
				{
					comps::ProviderKind::TakenHouse(_) => sleepyness_change = -500,
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
			}
			if let Some(kind) = plot_complete
			{
				let (provider, scenery_draw) = self
					.world
					.query_one_mut::<(&mut comps::Provider, &mut comps::SceneryDraw)>(provider_id)
					.unwrap();
				provider.kind = kind;
				if kind == comps::ProviderKind::Port
				{
					self.port = Some(provider_id);
				}
				scenery_draw.sprite = kind.get_sprite().to_string();
				self.world.remove_one::<comps::Plot>(provider_id)?;
			}

			println!("Work unit!");
		}
		for (provider_id, change) in providers_to_change
		{
			let mut provider = self.world.get::<&mut comps::Provider>(provider_id).unwrap();
			provider.num_occupants += change;
		}

		// Agent -> AgentDraw correspondence
		for (_, (agent, agent_draw, can_move)) in self
			.world
			.query::<(&comps::Agent, &mut comps::AgentDraw, &comps::CanMove)>()
			.iter()
		{
			agent_draw.visible = can_move.moving || agent.cur_provider.is_none();
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
				//for _ in 0..2
				//{
				//	spawn_agent(port_pos, &mut self.world, state, &mut self.rng)?;
				//}
			}
		}
		let mut houses_to_free = vec![];
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
				}
			}
		}
		for house_id in houses_to_free
		{
			let (provider, scenery_draw) = self
				.world
				.query_one_mut::<(&mut comps::Provider, &mut comps::SceneryDraw)>(house_id)
				.unwrap();
			provider.kind = comps::ProviderKind::EmptyHouse;
			scenery_draw.sprite = provider.kind.get_sprite().to_string();
		}

		// Remove dead entities
		to_die.sort();
		to_die.dedup();
		for id in to_die
		{
			println!("died {id:?}");
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
				match self.cursor_kind
				{
					CursorKind::BuildingPlacement(building_placement) =>
					{
						self.world.despawn(building_placement)?;
					}
					_ => (),
				}
				self.cursor_kind = CursorKind::Normal;
			}
			Event::MouseButtonDown { button, .. } =>
			{
				if button == 1
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
								spawn =
									Some((pixel_to_tile(position.pos), building_placement.kind));
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
									break;
								}
							}
							if let Some(id) = to_die
							{
								self.world.despawn(id)?;
							}
						}
						_ => (),
					}
					if let Some((position, kind)) = spawn
					{
						spawn_provider(position, kind, &mut self.world, state)?;
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

		for (_, (position, scenery_draw)) in self
			.world
			.query::<(&comps::Position, &comps::SceneryDraw)>()
			.iter()
		{
			let sprite = state.get_sprite(&scenery_draw.sprite).unwrap();

			sprite.draw(
				to_f32(position.pos - self.camera_pos),
				0,
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}
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
			}
			_ => (),
		}
		state.core.hold_bitmap_drawing(false);
		Ok(())
	}
}
