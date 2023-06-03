use crate::error::Result;
use crate::{astar, components as comps, game_state, sprite, utils};
use allegro::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

use std::collections::HashMap;

static TILE_SIZE: i32 = 16;

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

pub fn spawn_agent<T: Rng>(
	tile_pos: Point2<i32>, world: &mut hecs::World, state: &mut game_state::GameState, rng: &mut T,
) -> Result<hecs::Entity>
{
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::TilePath {
			tile_path: vec![tile_pos + Vector2::new(2, 0)],
		},
		comps::AgentDraw {
			sprite: "data/cat1.cfg".to_string(),
			visible: true,
		},
		comps::CanMove { moving: false },
		comps::Agent {
			time_to_think: 0.,
			time_to_work: 0.,
			cur_provider: None,
            house: None,
		},
	));
	Ok(entity)
}

pub fn spawn_provider(
	tile_pos: Point2<i32>, kind: comps::ProviderKind, world: &mut hecs::World,
	state: &mut game_state::GameState,
) -> Result<hecs::Entity>
{
	let sprite = match kind
	{
		comps::ProviderKind::EmptyHouse => "data/crystal1.cfg",
		comps::ProviderKind::TakenHouse(_) => "data/crystal1.cfg",
		comps::ProviderKind::Work => "data/crystal2.cfg",
	};
	let entity = world.spawn((
		comps::Position {
			pos: tile_to_pixel(tile_pos),
			dir: 0,
		},
		comps::SceneryDraw {
			sprite: sprite.to_string(),
		},
		comps::Provider {
			kind: kind,
			num_occupants: 0,
			max_occupants: 1,
		},
	));
	Ok(entity)
}

pub fn get_tile_path(
	from: Point2<i32>, to: Point2<i32>, size: usize, walkable: &[bool],
) -> Vec<Point2<i32>>
{
	let mut ctx = astar::AStarContext::new(size);
	ctx.solve(
		from,
		to,
		|pos| tile_to_idx(pos, size).map_or(true, |idx| !walkable[idx]),
		|pos| 1.,
	)
}

struct Map
{
	size: usize,
	terrain: Vec<bool>,
	world: hecs::World,
	rng: StdRng,
}

impl Map
{
	pub fn new(state: &mut game_state::GameState) -> Result<Self>
	{
		let size = 16;
		let mut terrain = Vec::with_capacity(size * size);
		let mut rng = StdRng::seed_from_u64(0);

		state.cache_sprite("data/terrain.cfg")?;
		state.cache_sprite("data/cat1.cfg")?;
		state.cache_sprite("data/crystal1.cfg")?;
		state.cache_sprite("data/crystal2.cfg")?;

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
					true
				};
				terrain.push(val)
			}
		}
		let mut world = hecs::World::new();
		let from = Point2::new(0, 0);
		//terrain[tile_to_idx(from, size).unwrap()] = true;
		let agent = spawn_agent(from, &mut world, state, &mut rng)?;
		let from = Point2::new(9, 0);
		let agent = spawn_agent(from, &mut world, state, &mut rng)?;

		spawn_provider(
			Point2::new(7, 7),
			comps::ProviderKind::EmptyHouse,
			&mut world,
			state,
		)?;
		spawn_provider(
			Point2::new(6, 2),
			comps::ProviderKind::EmptyHouse,
			&mut world,
			state,
		)?;
		spawn_provider(
			Point2::new(2, 6),
			comps::ProviderKind::Work,
			&mut world,
			state,
		)?;
		spawn_provider(
			Point2::new(4, 5),
			comps::ProviderKind::Work,
			&mut world,
			state,
		)?;

		Ok(Self {
			size: size,
			terrain: terrain,
			world: world,
			rng: rng,
		})
	}

	fn logic(&mut self, state: &mut game_state::GameState)
		-> Result<Option<game_state::NextScreen>>
	{
		// TilePath
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

		// Index the providers.
		let mut kind_to_providers = HashMap::new();
		for (id, (position, provider)) in self
			.world
			.query::<(&comps::Position, &comps::Provider)>()
			.iter()
		{
			if provider.num_occupants == provider.max_occupants
			{
				continue;
			}
			let mut providers = kind_to_providers
				.entry(provider.kind)
				.or_insert_with(Vec::new);
			providers.push((id, pixel_to_tile(position.pos)));
		}

		// Agent.
		let mut providers_to_change = vec![];
		let mut providers_to_work = vec![];
		let mut providers_to_house_claim = vec![];
		for (agent_id, (position, tile_path, can_move, agent)) in self
			.world
			.query::<(
				&comps::Position,
				&mut comps::TilePath,
				&mut comps::CanMove,
				&mut comps::Agent,
			)>()
			.iter()
		{
			if can_move.moving
			{
				continue;
			}
			if state.time() < agent.time_to_think
			{
				if state.time() > agent.time_to_work
				{
					if let Some(provider) = agent.cur_provider
					{
						providers_to_work.push(provider);
					}
					agent.time_to_work = state.time() + 1.;
				}
				continue;
			}
			agent.time_to_think = state.time() + 5. + self.rng.gen::<f64>();

			if let Some(provider) = agent.cur_provider.take()
			{
				providers_to_change.push((provider, -1));
			}

			let action = [0, 1, 2].choose(&mut self.rng).unwrap();

			match action
			{
				0 =>
				// Goof off
				{
					let cur_pos = pixel_to_tile(position.pos);
					let mut target = cur_pos
						+ Vector2::new(self.rng.gen_range(-3..=3), self.rng.gen_range(-3..=3));
					target.x = utils::clamp(target.x, 0, self.size as i32 - 1);
					target.y = utils::clamp(target.y, 0, self.size as i32 - 1);
					let cand_tile_path = get_tile_path(cur_pos, target, self.size, &self.terrain);
					tile_path.tile_path = cand_tile_path;
					can_move.moving = true;

					println!("Action:  Goof off");
				}
				_ =>
				{
					let mut work_options = vec![comps::ProviderKind::Work];
					if agent.house.is_none()
					{
						work_options.push(comps::ProviderKind::EmptyHouse);
					}
					else
					{
						work_options.push(comps::ProviderKind::TakenHouse(agent_id));
					}
					let kind = work_options.choose(&mut self.rng).unwrap();
					println!("Action: {kind:?}");
					if let Some(providers) = kind_to_providers.get(kind)
					{
						let mut provider_and_scores = Vec::with_capacity(providers.len());
						let mut total_score = 0.;
						for (provider_id, provider_pos) in providers
						{
							let dist = (to_f32(*provider_pos) - to_f32(position.pos)).magnitude();
							let score = 1. / (1. + dist);
							total_score += score;
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
								&self.terrain,
							);
							if !cand_tile_path.is_empty()
							{
								tile_path.tile_path = cand_tile_path;
								can_move.moving = true;
								providers_to_change.push((*provider_id, 1));
								agent.cur_provider = Some(*provider_id);
								if *kind == comps::ProviderKind::EmptyHouse
								{
									agent.house = Some(*provider_id);
									providers_to_house_claim.push((*provider_id, agent_id));
								}
								break;
							}
						}
					}
				}
			}
		}
		for (provider_id, agent_id) in providers_to_house_claim
		{
			let mut provider = self.world.get::<&mut comps::Provider>(provider_id).unwrap();
			provider.kind = comps::ProviderKind::TakenHouse(agent_id);
		}
		for provider_id in providers_to_work
		{
			println!("Work!");
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
		Ok(None)
	}

	fn input(
		&mut self, event: &Event, state: &mut game_state::GameState,
	) -> Result<Option<game_state::NextScreen>>
	{
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
					Point2::new((x * TILE_SIZE) as f32, (y * TILE_SIZE) as f32),
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
				to_f32(position.pos),
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
				to_f32(position.pos),
				variant,
				if agent_draw.visible
				{
					Color::from_rgb_f(1., 1., 1.)
				}
				else
				{
					Color::from_rgb_f(0.5, 0.5, 1.)
				},
				state,
			);
		}
		state.core.hold_bitmap_drawing(false);
		Ok(())
	}
}
