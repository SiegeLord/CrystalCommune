use crate::error::Result;
use crate::{astar, components as comps, game_state, sprite};
use allegro::*;
use na::{
	Isometry3, Matrix4, Perspective3, Point2, Point3, Quaternion, RealField, Rotation2, Rotation3,
	Unit, Vector2, Vector3, Vector4,
};
use nalgebra as na;
use rand::prelude::*;

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
		},
        comps::CanMove { moving: false }
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
					rng.gen_bool(0.7)
				};
				terrain.push(val)
			}
		}
        let from = Point2::new(1, 1);
        terrain[tile_to_idx(from, size).unwrap()] = true;

		let mut world = hecs::World::new();
		let agent = spawn_agent(from, &mut world, state, &mut rng)?;
		{
			let mut tile_path = world.get::<&mut comps::TilePath>(agent).unwrap();
            tile_path.tile_path = get_tile_path(from, Point2::new(8, 8), size, &terrain);
            println!("Here {:?}", tile_path.tile_path);
		}

		Ok(Self {
			size: size,
			terrain: terrain,
			world: world,
		})
	}

	fn logic(&mut self, state: &mut game_state::GameState)
		-> Result<Option<game_state::NextScreen>>
	{
		for (_, (position, can_move, tile_path)) in self
			.world
			.query::<(&mut comps::Position, &mut comps::CanMove, &mut comps::TilePath)>()
			.iter()
		{
			if tile_path.tile_path.is_empty()
			{
                can_move.moving = false;
				continue;
			}

			let target_tile = *tile_path.tile_path.last().unwrap();
			let target = tile_to_pixel(target_tile);
			if target.x > position.pos.x
			{
                can_move.moving = true;
				position.pos.x += 1;
                position.dir = 0;
			}
			if target.y > position.pos.y
			{
                can_move.moving = true;
				position.pos.y += 1;
                position.dir = 1;
			}
			if target.x < position.pos.x
			{
                can_move.moving = true;
				position.pos.x -= 1;
                position.dir = 2;
			}
			if target.y < position.pos.y
			{
                can_move.moving = true;
				position.pos.y -= 1;
                position.dir = 3;
			}
			if target == position.pos
			{
				tile_path.tile_path.pop().unwrap();
			}
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

		for (_, (position, can_move, agent_draw)) in self
			.world
			.query::<(&comps::Position, &comps::CanMove, &comps::AgentDraw)>()
			.iter()
		{
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
				Color::from_rgb_f(1., 1., 1.),
				state,
			);
		}

		state.core.hold_bitmap_drawing(false);
		Ok(())
	}
}
